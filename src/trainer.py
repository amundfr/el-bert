"""
This class takes a BERT model (e.g. BertBinaryClassification) and train,
 validation and test datasets and performs training and testing.
"""

import time
import torch

from typing import Optional, List
from transformers import AdamW
# from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler

from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia
from src.bert_model import BertMdEd, save_model_to_dir
from src.evaluator import Evaluator
from src.toolbox import format_time, print_training_stats


class ModelTrainer:
    def __init__(
                self,
                model: BertMdEd,
                train_dataloader: DataLoader,
                validation_dataloader: DataLoader,
                test_dataloader: DataLoader,
                ed_labels: List,
                dataset_to_doc: List,
                dataset_to_doc_pos: List,
                learning_rate: Optional[float] = 2e-5,
                epochs: Optional[int] = 3,
                gpu_index: Optional[int] = 0,
            ):
        """
        :param model: a BertEdMd model
        :param train_dataloader: a torch dataloader for training data
        :param validation_dataloader: a torch dataloader for validation data
        :param test_dataloader: a torch dataloader for test data
        :param ed_labels: a list of the QIDs of 'B' tokens with an embedding
            for each sequence
        :param dataset_to_doc: List of which document each data point is from
            Necessary to group split documents, and
            show accuracy over documents
        :param dataset_to_doc_pos: List of position of each data point sequence
            in tokenized origin document. Necessary to find overlapping
            sequences in evaluation.
        :param learning_rate: initial learning rate
        :param epochs: number of training epochs
        :param gpu_index: the index of the GPU to use for training
        """
        self.model = model

        # Use Cuda if Cuda enabled GPU is available
        # Using the GPU with the provided index
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_index))
            print('Using device:', torch.cuda.get_device_name(0))
        else:
            print('Using CPU')

        # Model is moved to device in-place, but tensors are not:
        # Source: https://discuss.pytorch.org/t/model-move-to-device-gpu/105620
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        self.ed_labels = ed_labels
        self.dataset_to_doc = dataset_to_doc
        self.dataset_to_doc_pos = dataset_to_doc_pos

        self.epochs = int(epochs)
        total_steps = len(self.train_dataloader) * self.epochs
        if self.epochs == 0:
            total_steps = 1
        print(f"Total training steps: {total_steps}")

        self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,   # start learning rate
                # weight_decay=0.01,  # weight decay (TODO: HPO)
                correct_bias=False,  # To stay closer to TF default
            )

        # Create the learning rate scheduler.

        # Same as BlackRock et al. default (following mail correspondence)
        # Linear LR scheduler. lr_lambda returns a
        #   factor so that LR = initial_LR * lr_lambda(epoch)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda e: 1 - e/total_steps,
            )
        # SDGR LR Scheduler:
        # T_0 = 1  # max(1, round(self.epochs/20))
        # self.scheduler = \
        #     torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         self.optimizer,
        #         T_0=T_0,  # iterations before first restart
        #         T_mult=2,
        #         eta_min=0,
        #         last_epoch=-1,
        #         verbose=False
        #     )

        # # OneCycle LR Scheduler:
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #         self.optimizer,
        #         max_lr=learning_rate,
        #         total_steps=total_steps
        #     )

        # # Cosine LR schedule:
        # self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #         self.optimizer,
        #         num_warmup_steps=0,  # warm start
        #         num_training_steps=total_steps,
        #         num_cycles=1  # round(self.epochs/8)  # try: self.epochs
        #     )

    def train(
                self,
                knowledgebase: Optional[KnowledgeBaseWikipedia] = None,
                early_stopping: Optional[bool] = True,
                train_update_freq: Optional[int] = 20,
                validation_update_freq: Optional[int] = 10,
                save_every_n_epochs: Optional[int] = None,
                checkpoint_save_dir: Optional[str] = None,
                documents_file: Optional[str] = None,
            ):
        """
        Train the model
        :param knowledgebase: KnowlegeBase object. If provided, get ED
            validation result each epoch, with candidate generation, if
            KB allows it
        :param early_stopping: Use early stopping ?
        :param train_update_freq: Progress feedback frequency
            in number of training batches
        :param validation_update_freq: Progress feedback frequency
            in number of validation batches
        :param save_every_n_epochs: The frequency of saving checkpoints
        :param checkpoint_save_dir: Path to directory to save model
            checkpoints after each epoch
        :param documents_file: file with documents for candidate generation
        :returns: training statistics
        """
        # Set default saving frequency, if none is given
        if checkpoint_save_dir and not save_every_n_epochs:
            save_every_n_epochs = round(self.epochs/4)
        if checkpoint_save_dir:
            save_every_n_epochs = max(1, save_every_n_epochs)

        if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                    ):
            # Evaluate the learning rate scheduler to know
            #   when to save checkpoints
            lrs = []
            for i in range(self.epochs):
                self.scheduler.step(i)
                lrs += [self.optimizer.param_groups[0]["lr"]]

        training_stats = []
        total_t0 = time.time()

        # If validation loss doesn't improve by more than this threshold,
        #   early stopping is triggered
        early_stopping_threshold = 0.0001
        # Number of epochs of patience with validation loss improvement
        #  under threshold before triggering early stopping
        early_stopping_patience = 5
        # Epoch with lowest validation loss during early stopping patience
        best_epoch = 0
        # Model with lowest validation loss during patience
        best_model = self.model.state_dict()

        print(f"\n   Training starts at {time.ctime(total_t0)}\n")

        # Iterate epochs
        for i_epoch in range(self.epochs):

            # Perform one full pass over the training set

            print(f"\n======== Epoch {i_epoch + 1} / {self.epochs} ========")
            print("\nTraining...")

            # Measure how long the training epoch takes.
            train_loss, train_duration, _, _, _, _ = \
                self.run_epoch('train', train_update_freq, i_epoch)

            # If save directory provided, and not first or last epoch
            if checkpoint_save_dir and i_epoch != self.epochs - 1:
                # If scheduler is SGDR, the LR is about to warm up
                if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                        ) and lrs[i_epoch] < lrs[i_epoch + 1]:
                    save_model_to_dir(
                            self.model,
                            checkpoint_save_dir
                        )
                if not isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                        ) and ((i_epoch + 1) % save_every_n_epochs) == 0:
                    save_model_to_dir(
                            self.model,
                            checkpoint_save_dir
                        )

            # Calculate the average loss over all of the batches.
            avg_train_loss = sum(train_loss) / len(train_loss)

            # Measure how long this epoch took.
            train_duration = format_time(train_duration)

            print("\nEpoch training statistics:")
            print(f"  Average training loss: {avg_train_loss:.4f}")
            print(f"  Training epoch took: {train_duration}")
            print(f"  Last LR: {self.scheduler.get_last_lr()[0]:.4e}")

            evaluation_result, _, val_duration, val_loss = \
                self.evaluate(
                        dataset='val',
                        knowledgebase=knowledgebase,
                        update_freq=validation_update_freq,
                        documents_file=documents_file,
                        use_fallback_for_empty_cs=False,
                    )
            md_f1 = evaluation_result[0]
            ed_f1 = evaluation_result[1]

            print("\nEpoch validation statistics:")
            val_duration = format_time(val_duration)
            print(f"       Duration: {val_duration}"
                  f"\n  Avgerage loss: {val_loss:.4f}"
                  f"\n    MD Micro F1: {md_f1*100:.4f}"
                  f"\n    ED Micro F1: {ed_f1*100:.4f}")

            # Record all statistics from this epoch.
            training_stats.append({
                    'epoch': i_epoch + 1,
                    'train loss': avg_train_loss,
                    'train time': train_duration,
                    'val loss': val_loss,
                    'val time': val_duration,
                    'val MD F1': float(100*md_f1),
                    'val ED F1': float(100*ed_f1),
                })

            # Early stopping if validation loss doesn't improve sufficiently
            if early_stopping and i_epoch > 1:
                delta_val_loss = \
                    training_stats[-2]['val loss'] \
                    - training_stats[-1]['val loss']

                # If no improvement, decrement patience
                if delta_val_loss < early_stopping_threshold:
                    early_stopping_patience += -1
                    # Check if this model is the best model
                    if val_loss < training_stats[best_epoch]['val loss']:
                        best_epoch = i_epoch
                        best_model = self.model.state_dict()

                    # If no more patience
                    if early_stopping_patience == 0:
                        print(f"Triggering early stopping after "
                              f"{i_epoch + 1} epochs. "
                              f"Best model is from epoch {best_epoch}.")
                        self.model = BertMdEd.load_state_dict(best_model)
                        break
                else:
                    # Reset patience
                    early_stopping_patience = 5

        print("\nTraining complete!")

        print(f"Total training took "
              f"{format_time(time.time() - total_t0)} (h:mm:ss)")

        print_training_stats(training_stats)

        return training_stats

    def run_epoch(
                self,
                dataset: str,
                feedback_frequency: Optional[int] = 10,
                i_epoch: Optional[int] = 1,
                train: bool = None,
            ):
        """
        Perform a training epoch
        :param dataset: is 'train', 'validation' or 'test'
        :param feedback_frequency: print progress after this many batches
        :param i_epoch: necessary for the learning rate scheduler
        :param train: if None (default), infer from dataset;
            if True, train model weights; if False, only evaluate.
        """
        t0 = time.time()
        total_loss = []
        feedback_msg = "  Batch {:>5,}  of  {:>5,}.    Elapsed: {}.    " + \
            "Avg loss last {}: {:.4f}"

        # Used if in evaluation mode (type 'validation' or 'test')
        e_md_logits = torch.Tensor()
        e_md_labels = torch.Tensor()
        e_ed_embeddings = torch.Tensor()
        e_ed_emb_labels = torch.Tensor()

        # Setup for different epoch modes
        if dataset == 'train':
            # Put the model into training mode
            dataloader = self.train_dataloader
            # If no other instructions, set to training mode
            if train is None:
                train = True
        elif dataset == 'val':
            # Put the model into evaluation mode
            dataloader = self.validation_dataloader
            # If no other instructions, set to evaluation mode
            if train is None:
                train = False
        elif dataset == 'test':
            # Put the model into evaluation mode
            dataloader = self.test_dataloader
            # If no other instructions, set to evaluation mode
            if train is None:
                train = False
        else:
            raise ValueError(f"Epoch dataset must be 'train', 'val' or"
                             f" 'test'. Got {dataset}.")

        # Set model to train
        if train is True:
            self.model.train()
            torch.set_grad_enabled(True)
        # Set model to evaluate
        elif train is False:
            # Workaround to make sure the dataloader is sequential for train
            if dataset == 'train':
                dataloader = DataLoader(
                    self.train_dataloader.dataset,
                    sampler=SequentialSampler(self.train_dataloader.dataset),
                    batch_size=self.train_dataloader.batch_size,
                )
            self.model.eval()
            torch.set_grad_enabled(False)

        # Run epoch
        for step, batch in enumerate(dataloader):
            # Progress update every few batches.
            if step % feedback_frequency == 0 and not step == 0:
                avg_loss = \
                    sum(total_loss[-feedback_frequency:])/feedback_frequency
                elapsed = format_time(time.time() - t0)
                print(feedback_msg.format(
                        step, len(dataloader), elapsed,
                        feedback_frequency, avg_loss
                    ))

            # Unpack this training batch from the dataloader
            #  and move to correct device and data type
            b_input_ids = \
                batch[0].to(device=self.device, dtype=torch.long)
            b_attention_mask = \
                batch[1].to(device=self.device, dtype=torch.long)
            b_token_type_ids = \
                batch[2].to(device=self.device, dtype=torch.long)
            b_md_labels = \
                batch[3].to(device=self.device, dtype=torch.float)
            b_ed_emb_labels = \
                batch[4].to(device=self.device, dtype=torch.float)
            b_labels = (b_md_labels, b_ed_emb_labels)

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass on the model
            outputs = self.model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels,
                    return_dict=True
                 )
            # Loss of the batch from the forward pass' loss function
            loss = outputs.loss
            # The IOB predictions
            logits = outputs.logits
            # The ED embeddings
            embeddings = outputs.embeddings
            # Add to total epoch loss
            total_loss += [loss.item()]

            # Take a training step, if training
            if train is True:
                # Perform a backward pass to calculate the gradients
                loss.backward()
                # Clip the norm of the gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Update the weights with the optimizer
                self.optimizer.step()
                # Tell the scheduler to update the learning rate
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
                        ):
                    self.scheduler.step(i_epoch + step / len(dataloader))
                elif self.scheduler is not None:
                    self.scheduler.step()

            # If evaluating, get predictions and labels
            else:
                b_md_logits = logits.detach().cpu()
                b_ed_embeddings = embeddings.cpu()
                b_md_labels = b_md_labels.cpu()
                b_ed_emb_labels = b_ed_emb_labels.cpu()

                e_md_logits = \
                    torch.cat((e_md_logits, b_md_logits), dim=0)
                e_ed_embeddings = \
                    torch.cat((e_ed_embeddings, b_ed_embeddings), dim=0)
                e_md_labels = \
                    torch.cat((e_md_labels, b_md_labels), dim=0)
                e_ed_emb_labels = \
                    torch.cat((e_ed_emb_labels, b_ed_emb_labels), dim=0)

        epoch_duration = time.time()-t0

        return total_loss, epoch_duration, e_md_logits, \
            e_ed_embeddings, e_md_labels, e_ed_emb_labels

    def evaluate(
                self,
                dataset: str,
                knowledgebase: KnowledgeBaseWikipedia,
                sample_n: Optional[int] = 0,
                update_freq: Optional[int] = 20,
                verbose: Optional[bool] = False,
                candidate_generation: bool = True,
                documents_file: str = '',
                use_fallback_for_empty_cs: str = True,
            ):
        """
        Evaluate the model with the requested dataset.
        Relies on mappings from data point to documents and mentions
        in order to group data points over mentions.

        :param dataset: is 'train', 'validation' or 'test'
        :param knowledgebase: KnowlegeBase object, needed to find
            entities from embeddings
        :param sample_n: Number of documents to randomly sample for evaluation
            if < 0, all documents are evaluated
        :param update_freq: progress feedback frequency
            in number of test batches
        :param verbose: if True, prints the evaluation result
        :param candidate_generation: if True, triggers candidate generation
        :param documents_file: necessary to use candidate generation
        :param use_fallback_for_empty_cs: if True, do similarity search with
            all entities when there are no candidates (slows down training)
        :returns: A tuple of:
            * Duration of the model forward pass (not evaluation itself)
            * Average loss of the evaulation forward pass
            * Micro F1 Score of Mention Detection
            * Micro F1 Score of Entity Disambiguation
        """
        evaluator, duration, loss = self.get_evaluator(
                knowledgebase, dataset, update_freq, documents_file
            )

        evaluation_result = evaluator.evaluation(
                sample_n,
                candidate_generation,
                use_fallback_for_empty_cs,
            )
        if verbose:
            print(evaluator.evaluation_str())

        return evaluation_result, evaluator, duration, loss

    def get_evaluator(
                self,
                knowledgebase: KnowledgeBaseWikipedia,
                dataset: str = 'test',
                update_freq: int = 20,
                documents_file: str = '',
            ):
        """
        Run evaluation on a dataset (train, val, test), and get an Evaluator
            object for evaluation
        :param knowledgebase: a KnowledgeBase object used by ED evaluation to
            find candidates and ED predictions
        :param dataset: a string of either 'test', 'val' or 'train' for which
            dataset to evaluate
        :param update_freq: Update frequency of the model evaluation
            in number of epochs
        :param documents_file: a file with documents for candidate generation
        :returns: an initialized Evaluator object for the given dataset
        """
        if dataset not in ['train', 'val', 'test']:
            raise ValueError(
                    f"Parameter 'dataset' in function 'evaluate' was '"
                    f"{dataset}'. Expected one of ['train', 'val', 'test']"
                )
        # Run the evaluation epoch
        loss, duration, md_logits, \
            ed_embeddings, md_labels, _ = \
            self.run_epoch(dataset, update_freq, train=False)
        # Average loss over batches.
        avg_loss = sum(loss) / len(loss)

        # Find the bounds of the current dataset in the sequential lists
        # Train dataset
        if dataset == 'train':
            # Make dataloader sample sequentially for this
            dataset = DataLoader(
                    self.train_dataloader.dataset,
                    sampler=SequentialSampler(self.train_dataloader.dataset),
                    batch_size=self.train_dataloader.batch_size,
                )
            data_start = 0
        # Val dataset
        elif dataset == 'val':
            dataset = self.validation_dataloader
            data_start = len(self.train_dataloader.dataset.indices)
        # Test dataset
        else:
            dataset = self.test_dataloader
            data_start = len(self.train_dataloader.dataset.indices) + \
                len(self.validation_dataloader.dataset.indices)
        data_end = data_start + len(md_labels)

        ed_label_vectors = torch.Tensor([])
        # Get the input_ids and the ed label vectors off of
        #   the correct dataloader
        for batch in dataset:
            ed_label_vectors = torch.cat((ed_label_vectors, batch[4]))

        ed_labels = self.ed_labels[data_start:data_end]
        docs = self.dataset_to_doc[data_start:data_end]
        pos = self.dataset_to_doc_pos[data_start:data_end]

        evaluator = Evaluator(
                md_preds=md_logits,
                md_labels=md_labels,
                ed_preds=ed_embeddings,
                ed_label_vectors=ed_label_vectors,
                ed_label_ids=ed_labels,
                docs=docs,
                positions=pos,
                knowledgebase=knowledgebase,
                documents_file=documents_file,
            )
        return evaluator, duration, avg_loss

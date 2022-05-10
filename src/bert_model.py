"""
Custom BERT model, extending BertPreTrainedModel

The BERT model has two heads: One for Mention Detection,
and one to create embeddings for Entity Disambiguation.
The heads have either no hidden layers (BertMdEdHeads)
or one hidden layer (BertMdEdHeadsWithHiddenLayer)

Requires:
A pretrained BERT model, or a reference to a pretrained Huggingface model
"""

import time
import torch

from typing import Optional, List, Tuple
from os.path import join, isdir
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.configuration_bert import BertConfig

from src.input_data_generator import InputDataGenerator


class BertMdEdOutput(ModelOutput):
    """
    Custom output dictionary,
        used if return_dict = True in model's forward
    """
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class BertMdEdHeads(torch.nn.Module):
    """
    Custom output head for model:
        output layer from input logit embeddings
        to 3 MD labels (Inside, Outside, Beginning),
        and a vector of 100 dimensions for ED embeddings
        and a Tanh activation function for the ED output
    """
    def __init__(self, config: BertConfig):
        """
        :param config: the BertConfig object of a BertMdEd model
        """
        super().__init__()
        # Number of classes for Mention Detection
        num_md_labels = 3
        # The Wikipedia2vec embedding dimension
        self.embedding_dim = 100

        # # Only one output layer:
        self.md_classifier = \
            torch.nn.Linear(config.hidden_size, num_md_labels)
        # For projection to the Wikidata2vec space
        self.ed_projection = \
            torch.nn.Linear(config.hidden_size, self.embedding_dim)
        self.ed_activation = torch.nn.Tanh()

    def forward(self, logits):
        """
        :param logits: a tensor of shape n_tokens * hidden_size
        :returns: a tuple of: MD predictions and ED embeddings for each token
        """
        md_pred = self.md_classifier(logits)
        ed_pred = self.ed_projection(logits)
        ed_pred = self.ed_activation(ed_pred)
        return md_pred, ed_pred


class BertMdEdHeadsWithHiddenLayer(torch.nn.Module):
    """
    Custom output head for model:
        hidden feed-forward layer for both MD and ED,
        output layers:
            MD labels: Inside, Outside, Beginning,
            ED embedding: 100 dimensions for ED embeddings,
                with Tanh activation for the ED output
    """
    def __init__(self, config: BertConfig):
        """
        :param config: the BertConfig object of a BertMdEd model
        """
        super().__init__()
        # Number of classes for Mention Detection
        num_md_labels = 3
        # The Wikipedia2vec embedding dimension
        self.embedding_dim = 100

        # Extra hidden output layer for each head:
        # For Mention Detection (labels {I, O, B})
        self.md_classifier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(
                    config.hidden_size,
                    eps=config.layer_norm_eps
                ),
            torch.nn.Linear(config.hidden_size, num_md_labels),
        )
        # For Entity Disambiguation (labels {I, O, B})
        self.ed_projection = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(
                    config.hidden_size,
                    eps=config.layer_norm_eps
                ),
            torch.nn.Linear(config.hidden_size, self.embedding_dim),
        )

        self.ed_activation = torch.nn.Tanh()

    def forward(self, logits):
        """
        :param logits: a tensor of shape n_tokens * hidden_size
        :returns: a tuple of: MD predictions and ED embeddings for each token
        """
        md_pred = self.md_classifier(logits)
        ed_pred = self.ed_projection(logits)
        ed_pred = self.ed_activation(ed_pred)
        return md_pred, ed_pred


class BertMdEd(BertPreTrainedModel):
    def __init__(
                self,
                config: BertConfig,
                hidden_output_layers: Optional[bool] = True,
                dropout_after_bert: Optional[bool] = False,
                loss_lambda: Optional[float] = 0.01,
            ):
        """
        :param config: a BertConfig object
        :param hidden_output_layers: if True, uses output heads with
            hidden layers. If False, no hidden layers in output heads
        :param dropout_after_bert: if True, uses dropout after BERT with
            default probability 0.1. If False, no dropout after BERT
        :param loss_lambda: the Lambda hyperparameter of the joint loss
        """
        super().__init__(config)

        # The BertModel without pooling layer
        self.bert = BertModel(config, add_pooling_layer=False)

        if 'use_dropout' in self.config.to_dict():
            self.use_dropout = self.config.use_dropout
        else:
            self.use_dropout = dropout_after_bert
            self.config.use_dropout = self.use_dropout

        self.dropout = None
        if self.use_dropout:
            # Dropout probability of BERT output
            dropout_prob = config.hidden_dropout_prob
            self.dropout = torch.nn.Dropout(dropout_prob)

        # Output heads
        if 'hidden_output_layers' in self.config.to_dict():
            hidden_output_layers = self.config.hidden_output_layers
        else:
            hidden_output_layers = hidden_output_layers
            self.config.hidden_output_layers = hidden_output_layers

        if hidden_output_layers:
            self.cls = BertMdEdHeadsWithHiddenLayer(config)
        else:
            self.cls = BertMdEdHeads(config)

        # Mention Detection loss
        self.md_loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=InputDataGenerator.IOB_LABEL['None'],
                reduction='mean'  # Note: 'mean' is the default
            )
        # Entity Disambiguation loss
        # margin=1.0 makes label '-1' not count (used for no embedding)
        self.ed_loss_fn = torch.nn.CosineEmbeddingLoss(
                margin=1.0,  # To ignore all non-labeled embeddings
                reduction='mean'  # Note: 'mean' is the default
            )

        # Weighting hyperparameter for the combined loss function (0.1 default)
        if 'loss_lambda' in self.config.to_dict():
            self.loss_lambda = self.config.loss_lambda
        else:
            self.loss_lambda = loss_lambda
            self.config.loss_lambda = self.loss_lambda

        self.init_weights()

    def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
            ):
        """
        Forward function of the model
        :param input_ids: Input token IDs, tokenized by a Bert Tokenizer
        :param token_type_ids: A sequence with True for tokens in sequence A
        :param attention_mask: A sequence with True for non-padding tokens
        :param labels: a tuple of (MD labels, ED labels)
        :param return_dict: if True, returns a BertMdEdOutput return dict
        :returns: a BertMdEdOutput return dict if return_dict=True,
            a tuple of loss and logits otherwise
        """
        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # Use sequence output
        bert_output = outputs[0]
        if self.use_dropout:
            bert_output = self.dropout(bert_output)
        # MD and ED predictions:
        logits, embeddings = self.cls(bert_output)

        loss = None
        if labels is not None:
            md_labels = labels[0]
            ed_labels = labels[1]

            # MD Loss
            md_loss = self.md_loss_fn(
                    logits.view(-1, logits.shape[-1]),
                    md_labels.view(-1).to(dtype=torch.long)
                )

            # Filter the ED labels and embeddings
            # Binary mask of tokens with a label entity vector
            non_zero = torch.logical_not(
                    torch.all(ed_labels == torch.zeros_like(ed_labels), dim=-1)
                ).view(-1)
            # Set tokens without label to -1 to ignore,
            embedding_target = torch.Tensor(non_zero.shape).fill_(-1)
            #  and to 1 for tokens with embeddings labels
            embedding_target[non_zero] = torch.Tensor([1.])
            # ED Loss
            ed_loss = self.ed_loss_fn(
                    embeddings.view(-1, self.cls.embedding_dim),
                    ed_labels.view(-1, self.cls.embedding_dim),
                    embedding_target.to(device=embeddings.get_device())
                )
            loss = \
                self.loss_lambda * md_loss + (1 - self.loss_lambda) * ed_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return BertMdEdOutput(
            loss=loss,
            logits=logits,
            embeddings=embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_layers(self, param_idx: List):
        """
        Freeze BERT layers at provided indices to not train them
        :param param_idx: list of indices of layers to be frozen
        """
        module_params = list(self.named_parameters())
        for param in (module_params[p] for p in param_idx):
            param[1].requires_grad = False

    def freeze_n_transformers(self, n: int = 11):
        """
        Freeze the provided number of encoders in the BERT architecture
         to not train them
        :param n: number of encoders to freeze
        """
        n = min(n, 12)
        n_emb_layers = 5
        n_layers_in_transformer = 12
        emb_layers = list(range(n_emb_layers))
        encoder_layers = list(
                range(
                        n_emb_layers,
                        n_emb_layers + n * n_layers_in_transformer
                    )
            )
        self.freeze_layers(emb_layers + encoder_layers)

    def freeze_bert(self):
        """
        Freezes all layers in BERT from training,
            allowing only training of output heads
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad = False

    def set_loss_lambda(self, loss_lambda: float):
        """
        Set the loss Lambda parameter
        """
        self.loss_lambda = loss_lambda


def load_bert_from_file(model_path: str, **kwargs):
    """
    Load a BertMdEd model from file
    :param model_path: path to directory with BertMdEd
        model and config file
    :param kwargs: any remaining keyword arguments passed to model's __init__
    :returns: BertMdEd model
    """
    if not isdir(model_path):
        raise FileNotFoundError(f"No BERT model at directory {model_path}.")
    model = BertMdEd.from_pretrained(model_path, **kwargs)
    return model


def save_model_to_dir(model: BertMdEd, model_dir: str):
    """
    Write a BertMdEd model to a directory
    :param model: a BertMdEd model
    :param model_path: path to destination directory for the model
    """
    sub_dir = "saved_" + time.strftime('%Y%m%d_%H%M', time.gmtime(time.time()))
    model_dir = join(model_dir, sub_dir)
    print(f"Saving model to directory: {model_dir}")
    model.save_pretrained(model_dir)
    return model_dir

"""
This class makes dataloaders and splits the dataset.
Takes the output tensors of InputDataGenerator as input.

Particularly useful is the classmethod load(directory_path)
"""

import torch
from typing import List, Tuple
from os.path import isfile, join, isdir
from os import mkdir
from torch.utils.data import TensorDataset, Subset, \
        DataLoader, RandomSampler, SequentialSampler


class DatasetGenerator:
    # Default file names for writing to and reading from files
    file_names = [
        'input_ids.pt',
        'attention_mask.pt',
        'token_type_ids.pt',
        'md_labels.pt',
        'ed_embeddings.pt',
        'ed_labels.tsv',
        'doc_ind_and_pos.csv'
    ]

    def __init__(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                md_labels: torch.Tensor,
                ed_embeddings: torch.Tensor,
                ed_labels: List,
                doc_ind_and_pos: List[Tuple]
            ):
        """
        :param input_ids: Tensor with input_ids
        :param attention_mask: Tensor with attention_mask
        :param token_type_ids: Tensor with token_type_ids
        :param md_labels: Tensor with one-hot IOB Mention Detection labels
        :param ed_embeddings: Tensor with entity embeddings for annotated
            entities in the dataset
        :param ed_labels: list of the QIDs of 'B' tokens with an embedding
        :param doc_ind_and_pos: which doc index each datapoint comes from,
            and their position in tokenized origin document
        """
        # The data tensors
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.md_labels = md_labels
        self.ed_embeddings = ed_embeddings
        self.ed_labels = ed_labels

        # List with document index of each sequence/data point
        self.doc_indices = [d[0] for d in doc_ind_and_pos]
        # List of position of each sequence in tokenized origin document
        #  ! without [CLS], [SEP] and [PAD] tokens !
        self.doc_pos = [d[1] for d in doc_ind_and_pos]

    def split_conll_default(
                self,
                batch_size: int = 32,
            ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits the loaded dataset sequentially into the default split
            for the AIDA-CoNLL dataset
        :param batch_size: the batch size for the resulting dataloaders
        :returns: a tuple of dataloaders for train, validation, test
        """
        ratios = [0.6791, 0.1552, 0.1657]
        return self.split_by_ratio(ratios, batch_size)

    def split_by_ratio(
                self,
                split_ratios: List[int] = [97, 1, 2],
                batch_size: int = 32,
            ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits the dataset sequentially in given ratios to
         train, validation and test subsets.
        Splits ratios are on document index,
         so the exact ratio of the subsets may not be as expected.

        :param split_ratios: the ratios of train, validation, and test
            respectively, as a list of three ratio values
        :param batch_size: the batch size used for the Dataloaders
        :returns: tuple of three torch Dataloader objects with
            training, validation and test data
        """
        dataset = TensorDataset(
                self.input_ids,
                self.attention_mask,
                self.token_type_ids,
                self.md_labels,
                self.ed_embeddings
            )

        # Number of source docs
        n_docs = len(set(self.doc_indices))

        min_doc_id = min(self.doc_indices)
        # ratio of training data, base 1
        train_ratio = split_ratios[0] / sum(split_ratios)
        end_doc_train = round(train_ratio * n_docs) - 1 + min_doc_id
        # ratio of validation data, base 1
        val_ratio = split_ratios[1] / sum(split_ratios)
        end_doc_val = end_doc_train + round(val_ratio * n_docs)
        # Find the indices for each subset in the dataset
        end_train = len(self.doc_indices) \
            - self.doc_indices[::-1].index(end_doc_train)
        end_val = len(self.doc_indices) \
            - self.doc_indices[::-1].index(end_doc_val)

        train_indices = list(range(0, end_train))
        val_indices = list(range(end_train, end_val))
        test_indices = list(range(end_val, len(self.doc_indices)))
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Random sampling of training data
        train_dataloader = DataLoader(
                train_dataset,
                sampler=RandomSampler(train_dataset),
                batch_size=batch_size
            )
        # Sequential sampling of validation data
        validation_dataloader = DataLoader(
                val_dataset,
                sampler=SequentialSampler(val_dataset),
                batch_size=batch_size
            )
        # Sequential sampling of test data
        test_dataloader = DataLoader(
                test_dataset,
                sampler=SequentialSampler(test_dataset),
                batch_size=batch_size
            )

        return train_dataloader, validation_dataloader, test_dataloader

    def save(self, directory: str):
        """
        Writes all data to files for later use.
            tensors to .pt files,
            ed_labels to .tsv and
            document indices and positions to .csv
        Can be loaded with class function DatasetGenerator.load(directory)
        :params directory: destination directory
        """
        # Make directory if it doesn't exist
        if not isdir(directory):
            mkdir(directory)
        # List of the tensors to save (aligned with file_names)
        tensors_to_save = [
                self.input_ids.to(dtype=torch.short),
                self.attention_mask.to(dtype=torch.bool),
                self.token_type_ids.to(dtype=torch.bool),
                self.md_labels.to(dtype=torch.int8),
                self.ed_embeddings
            ]

        for tensor, file in zip(tensors_to_save, self.file_names[:5]):
            torch.save(tensor, join(directory, file))
        # Write self.ed_labels
        with open(join(directory, self.file_names[5]), 'w') as file:
            for ed_labels_seq in self.ed_labels:
                file.write('\t'.join(str(lab) for lab in ed_labels_seq) + '\n')
        # Write self.doc_indices and self.doc_pos to common file
        with open(join(directory, self.file_names[6]), 'w') as file:
            file.write(','.join(str(i) for i in self.doc_indices))
            file.write('\n')
            file.write(','.join(f"{p[0]} {p[1]}" for p in self.doc_pos))

    @classmethod
    def load(cls, directory: str):
        """
        Reads a previously saved directory of .pt files generated by the
            save() function, and creates a DatasetGenerator object
        :param vectors_dir: directory with files from save()
        :returns: a loaded DatasetGenerator object
        """
        if not all([isfile(join(directory, f)) for f in cls.file_names]):
            raise FileNotFoundError(
                    f"Could not find all files in directory '{directory}'. "
                    f"First do dataset_generator.save({directory})"
                )

        # Read the tensors from default file names
        tensors = [torch.load(join(directory, f)) for f in cls.file_names[:5]]
        tensors[0] = tensors[0].to(dtype=torch.short)
        tensors[1] = tensors[1].to(dtype=torch.bool)
        tensors[2] = tensors[2].to(dtype=torch.bool)
        tensors[3] = tensors[3].to(dtype=torch.int8)

        # Read the ed_labels lists
        with open(join(directory, cls.file_names[5]), 'r') as file:
            ed_labels = [line.strip().split('\t') for line in file]
            # Convert string "None" to type None
            ed_labels = [
                    [None if lab == 'None' else lab for lab in sequence]
                    for sequence in ed_labels
                ]
        # Read the doc_ids and doc_pos lists
        with open(join(directory, cls.file_names[6]), 'r') as file:
            doc_ids_line = next(file)
            doc_pos_line = next(file)
            doc_ids = [int(i) for i in doc_ids_line.split(',')]
            doc_pos = [
                    (int(p.split(' ')[0]), int(p.split(' ')[1]))
                    for p in doc_pos_line.split(',')
                ]
            doc_ind_and_pos = list(zip(doc_ids, doc_pos))

        dataset_generator = DatasetGenerator(
                *tensors, ed_labels, doc_ind_and_pos
            )
        return dataset_generator

import unittest
from torch import Tensor
from os import remove, rmdir
from os.path import join

from src.dataset_generator import DatasetGenerator


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        # shape: (n_data × input_dim)
        input_ids = Tensor([1]*11).unsqueeze(1)
        attention_mask = input_ids
        token_type_ids = input_ids
        md_labels = input_ids
        ed_embeddings = input_ids
        ed_labels = [None] * 11
        ed_labels[5:7] = ['Cho_Oyu', 'Maputo']
        ed_labels = [ed_labels]

        doc_indices = [1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6]
        doc_pos = [
                (1, 2), (2, 5), (5, 7), (7, 9),
                (1, 2), (2, 3), (1, 2), (2, 3),
                (1, 2), (1, 2), (1, 2)
            ]
        doc_ind_and_pos = list(zip(doc_indices, doc_pos))

        self.subject = DatasetGenerator(
                input_ids,
                attention_mask,
                token_type_ids,
                md_labels,
                ed_embeddings,
                ed_labels,
                doc_ind_and_pos
            )

    def tearDown(self):
        # Get rid of test subject
        self.subject = None

    def test_split_by_ratio(self):
        """Test splitting dataset by ratios"""
        expected_train_len = 4
        expected_val_len = 4
        expected_test_len = 3
        result = self.subject.split_by_ratio([1, 2, 3], 1)

        expected_train_len_2 = 2
        expected_val_len_2 = 2
        expected_test_len_2 = 2
        result_2 = self.subject.split_by_ratio([1, 2, 3], 2)

        self.assertEqual(len(result[0]), expected_train_len)
        self.assertEqual(len(result[1]), expected_val_len)
        self.assertEqual(len(result[2]), expected_test_len)
        self.assertEqual(len(result_2[0]), expected_train_len_2)
        self.assertEqual(len(result_2[1]), expected_val_len_2)
        self.assertEqual(len(result_2[2]), expected_test_len_2)

    def test_split_conll_default(self):
        """
        Check that the DatasetGenerator uses correct AIDA-CoNLL Splits
        """
        input_tensor = Tensor([1]*1393).unsqueeze(1)
        doc_ind_pos = list(zip(range(1393), [0]*1393))

        subject = DatasetGenerator(
                input_tensor,
                input_tensor,
                input_tensor,
                input_tensor,
                input_tensor,
                input_tensor,
                doc_ind_pos,
            )
        dataloaders = subject.split_conll_default(batch_size=2000)
        train_len = sum(len(batch[0]) for batch in dataloaders[0])
        val_len = sum(len(batch[0]) for batch in dataloaders[1])
        test_len = sum(len(batch[0]) for batch in dataloaders[2])
        self.assertEqual(train_len, 946)
        self.assertEqual(val_len, 216)
        self.assertEqual(test_len, 231)

    def test_dump_load(self):
        """Test dumping to file and reading from file"""
        result = None
        dump_dir = 'test/dump_dir'
        self.subject.save(dump_dir)
        result = DatasetGenerator.load(dump_dir)
        for file in DatasetGenerator.file_names:
            remove(join(dump_dir, file))
        rmdir(dump_dir)

        self.assertEqual(result.doc_indices, self.subject.doc_indices)

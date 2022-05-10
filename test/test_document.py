import unittest
import torch
from configparser import ConfigParser
from os import path
# from src.document import Document, EntityLabel, EntityPrediction
from src.input_data_generator import InputDataGenerator
from src.document import get_document_from_sequences
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia


class TestDocument(unittest.TestCase):
    def setUp(self):
        """
        Test case: Five sequences from the same document, with overlaps.
        """
        config = ConfigParser()
        config.read('config.ini')

        kb_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']
        w2vec_file = \
            path.join(kb_dir, config['KNOWLEDGE BASE']['Wikipedia2Vec File'])

        # Knowledgebase to test ED prediction
        self.kb = KnowledgeBaseWikipedia(
                w2vec_file,
            )

        positions = [(0, 3), (3, 6), (5, 8), (6, 9), (8, 9)]
        self.input_ids = torch.Tensor([
                [101, 6435, 6435, 6435, 102],
                [101, 6435, 6435, 6435, 102],
                [101, 6435, 6435, 6435, 102],
                [101, 6435, 6435, 6435, 102],
                [101, 6435, 102, 0, 0]
            ]).to(dtype=torch.int64)

        self.ld = InputDataGenerator.IOB_LABEL
        ld = self.ld
        # First and last is 'O' for [CLS] and [SEP]
        self.md_labels = torch.Tensor([
                [ld['None'], ld['B'],    ld['I'],    ld['I'],    ld['None']],
                [ld['None'], ld['None'], ld['None'], ld['O'],    ld['None']],
                [ld['None'], ld['O'],    ld['B'],    ld['I'],    ld['None']],
                [ld['None'], ld['B'],    ld['I'],    ld['O'],    ld['None']],
                [ld['None'], ld['O'],    ld['None'], ld['None'], ld['None']]
            ])
        self.md_preds = torch.zeros((5, 5, 3))
        # All correct on first sequence
        self.md_preds[0, 0, :] = torch.Tensor([0.3, 0.3, 0.3])
        self.md_preds[0, 1, ld['B']] = 0.7  # Correct
        self.md_preds[0, 2, ld['I']] = 0.8  # Correct
        self.md_preds[0, 3, ld['I']] = 0.8  # Correct
        self.md_preds[0, 4, :] = torch.Tensor([0.3, 0.3, 0.3])
        # Second sequence, trying to predict where there is "None"
        # This shouldn't cause a penalty
        self.md_preds[1, 0, ld['O']] = 0.7  # Label 'None': Pred irrelevant
        self.md_preds[1, 1, ld['B']] = 0.9  # Label 'None': Pred irrelevant
        self.md_preds[1, 2, ld['I']] = 0.9  # Label 'None': Pred irrelevant
        self.md_preds[1, 3, ld['O']] = 0.8  # Correct, tied with [2, 1]
        self.md_preds[1, 4, ld['O']] = 0.7  # Label 'None': Pred irrelevant
        # Third sequence, overlaps 1 with previous and 2 with next
        self.md_preds[2, 0, :] = torch.Tensor([0.3, 0.3, 0.3])
        self.md_preds[2, 1, ld['O']] = 0.8  # Correct, tied with [1, 3]
        self.md_preds[2, 2, ld['B']] = 0.9  # Correct, wins tie with [3, 1]
        self.md_preds[2, 3, ld['I']] = 0.7  # Correct, loses tie with [3, 2]
        self.md_preds[2, 4, :] = torch.Tensor([0.3, 0.3, 0.3])
        # Fourth sequence, tie breaking with previous
        self.md_preds[3, 0, :] = torch.Tensor([0.3, 0.3, 0.3])
        self.md_preds[3, 1, ld['O']] = 0.8  # Wrong, loses tie with [2, 2]
        self.md_preds[3, 2, ld['B']] = 0.9  # Wrong, wins tie with [2, 3]
        self.md_preds[3, 3, ld['I']] = 0.8  # Wrong, wins tie with [4, 1]
        self.md_preds[3, 4, :] = torch.Tensor([0.3, 0.3, 0.3])
        # Fifth sequence, loses tie and has [PAD]. In no way relevant!
        self.md_preds[4, 0, :] = torch.Tensor([0.3, 0.3, 0.3])
        self.md_preds[4, 1, ld['O']] = 0.5  # Correct, but loses to [3, 3]
        self.md_preds[4, 2, ld['O']] = 0.9  # Pad, irrelevant
        self.md_preds[4, 3, ld['I']] = 0.8  # Pad, irrelevant
        self.md_preds[4, 4, :] = torch.Tensor([0.3, 0.3, 0.3])

        self.ed_labels = [[None for _ in range(5)] for _ in range(5)]
        # With labels at relevant positions
        self.ed_labels[0][1] = 'Nestor Makhno'
        self.ed_labels[2][2] = 'Fyodor Dostoyevsky'
        self.ed_labels[3][1] = 'Fyodor Dostoyevsky'

        entity_embedding_size = 100
        self.ed_emb_labels = \
            torch.zeros((5, 5, entity_embedding_size))
        self.ed_emb_labels[0, 1, :] = \
            torch.Tensor(self.kb.get_entity_vector('Nestor Makhno'))
        self.ed_emb_labels[2, 2, :] = \
            torch.Tensor(self.kb.get_entity_vector('Fyodor Dostoyevsky'))
        self.ed_emb_labels[3, 1] = \
            self.ed_emb_labels[2, 2]  # 'B', overlap with [2, 2]

        self.ed_preds = self.ed_emb_labels.clone().detach()
        # Wrong prediction, wins over correct in [3, 1]
        self.ed_preds[2, 2, :] = \
            torch.Tensor(self.kb.get_entity_vector('The Idiot'))

        self.subject = get_document_from_sequences(
                doc_id=1,
                positions=positions,
                md_preds=self.md_preds,
                md_labels=self.md_labels,
                ed_preds=self.ed_preds,
                ed_label_vectors=self.ed_emb_labels,
                ed_label_ids=self.ed_labels,
                doc_text=7*['Finland'],
            )
        self.subject.ed_prediction(self.kb)

    def tearDown(self):
        super().tearDown()
        self.kb = None
        self.subject = None
        self.config = None

    def test_get_document_from_sequences(self):
        """
        Check that get_document_from_sequences returns the expected Document
        """
        # Number of WordPiece tokens after overlap is resolved
        expected_len = 7
        self.assertEqual(expected_len, self.subject.doc_len)
        self.assertEqual(
                [3, 4],
                self.subject.none_pos
            )

    def test_parse_labels_parse_predictions(self):
        """
        Assert results of the _parse_labels and _parse_predictions functions
            that run during Document's initialization
        """
        self.assertEqual(
                [(0, 2), (4, 5)],
                list(self.subject.labels.keys())
            )
        self.assertEqual(
                [(0, 2), (4, 4), (5, 6)],
                list(self.subject.predictions.keys())
            )

    def test_get_doc_text_span(self):
        """
        Test the get_doc_text_span
        """
        n_tokens = self.subject.doc_len + len(self.subject.none_pos)
        token_span = (0, n_tokens)
        self.assertEqual(
                (0, self.subject.doc_len),
                self.subject.get_doc_text_span(token_span)
            )

    def test_md_prediction(self):
        """
        Check that md_prediction sets correct values for labels and predictions
        """
        self.assertEqual(
                [True, False, False],
                [pred.correct_md for pred in self.subject.predictions.values()]
            )
        self.assertEqual(
                [True, False],
                [label.predicted_md for label in self.subject.labels.values()]
            )

    def test_get_md_stats(self):
        """Check that MD predictions and labels are correctly sorted"""
        expected_tp = 1
        expected_fp = 2
        expected_fn = 1
        self.assertEqual(
                (expected_tp, expected_fp, expected_fn),
                self.subject.get_md_stats()
            )

    def test_ed_prediction(self):
        """
        Check if ed_prediction sets correct values for labels and predictions
        """
        expected_correct_eds = [True, False, False]
        expected_predicted_eds = [True, False]

        # No candidate generation
        preds = self.subject.predictions.values()
        labels = self.subject.labels.values()
        correct_eds = [pred.correct_ed for pred in preds]
        predicted_eds = [label.predicted_ed for label in labels]

        self.assertEqual(expected_correct_eds, correct_eds)
        self.assertEqual(expected_predicted_eds, predicted_eds)

    def test_get_ed_stats(self):
        """
        Check that the get_ed_stats function sorts
            labels and predictions correctly
        """
        expected_tp = 1
        expected_fp = 2
        expected_fn = 1
        expected_not_in_kb = 0
        expected_not_in_cand = 0
        expected_no_cand = 0
        expected_return = (expected_tp, expected_fp, expected_fn,
                           expected_not_in_kb, expected_not_in_cand,
                           expected_no_cand)
        result = self.subject.get_ed_stats()
        self.assertEqual(expected_return, result)

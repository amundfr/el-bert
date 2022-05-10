import unittest
import torch
from configparser import ConfigParser
from src.bert_model import BertMdEd, BertMdEdHeadsWithHiddenLayer
import transformers
transformers.logging.set_verbosity_error()


class TestBertModel(unittest.TestCase):
    def setUp(self):
        """
        Test case: Five sequences from the same document, with overlaps.
        """
        config = ConfigParser()
        config.read('config.ini')
        bert_model_id = config['MODEL']['Model ID']
        loss_lambda = config.getfloat('TRAINING', 'Loss Lambda')
        hidden_output_layers = \
            config.getboolean('MODEL', 'Hidden Output Layers')
        dropout_after_bert = config.getboolean('MODEL', 'Dropout After BERT')
        self.subject = BertMdEd.from_pretrained(
                bert_model_id,
                hidden_output_layers=hidden_output_layers,
                dropout_after_bert=dropout_after_bert,
                loss_lambda=loss_lambda,
            )

    def testInit(self):
        """
        Test that the model has the expected settings and default
            values after initialization
        """
        self.assertIsInstance(self.subject.bert, transformers.BertModel)
        self.assertFalse(self.subject.use_dropout)
        self.assertIsNone(self.subject.dropout)
        self.assertTrue(self.subject.config.hidden_output_layers)
        self.assertIsInstance(self.subject.cls, BertMdEdHeadsWithHiddenLayer)

    def testLoss(self):
        """
        Test the outputs of ED and MD loss functions
        """
        md_logits = torch.Tensor([[-2, 2, -2, -2], [2, -2, -2, -2]])
        md_labels = torch.Tensor([1, 1]).to(dtype=torch.long)
        md_expected = 2.0535
        md_result = self.subject.md_loss_fn(md_logits, md_labels)
        self.assertEqual(md_expected, round(md_result.item(), 4))

        ed_pred = torch.Tensor([[0, 1]]).to(dtype=torch.long)
        ed_label = torch.Tensor([[1, 0]]).to(dtype=torch.long)
        ed_expected = 1
        ed_result = self.subject.ed_loss_fn(
                ed_pred, ed_label, torch.Tensor([1])
            )
        self.assertEqual(ed_expected, ed_result)

        ed_pred = torch.Tensor([[0, 1]]).to(dtype=torch.long)
        ed_label = torch.Tensor([[0, 1]]).to(dtype=torch.long)
        ed_expected = 0
        ed_result = self.subject.ed_loss_fn(
                ed_pred, ed_label, torch.Tensor([1])
            )
        self.assertEqual(ed_expected, ed_result)

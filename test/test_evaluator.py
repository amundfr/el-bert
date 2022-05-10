import unittest
import torch
from configparser import ConfigParser
from src.dataset_generator import DatasetGenerator
from src.input_data_generator import InputDataGenerator
from src.evaluator import Evaluator
from src.toolbox import get_knowledge_base
import warnings


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """
        Test case: Five sequences from the same document, with overlaps.
        """
        # Ignore Runtime Warnings from Wikipedia2vec
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self.config = ConfigParser()
        self.config.read('config.ini')

        kb = get_knowledge_base(self.config, wiki='pedia', with_cg=False)

        n_docs = 5
        seq_len = 9
        input_ids = torch.Tensor(n_docs * [[
                #   Finland,                     Northern, Europe
                101, 6435, 2003, 1037, 2406, 1999, 2642, 2885, 102
            ]]).to(dtype=torch.int64)

        # Get the label ID to vector mapping from InputDataGenerator
        self.ld = InputDataGenerator.IOB_LABEL
        ld = self.ld
        md_preds = torch.nn.functional.one_hot(torch.Tensor([
                [ld['O'], ld['B'], ld['O'], ld['O'], ld['O'], ld['O'],
                 ld['B'], ld['I'], ld['O']],  # Correct
                [ld['O'], ld['B'], ld['B'], ld['O'], ld['O'], ld['O'],
                 ld['B'], ld['I'], ld['O']],  # 1 False Positive
                [ld['O'], ld['O'], ld['O'], ld['O'], ld['O'], ld['O'],
                 ld['B'], ld['I'], ld['O']],  # 1 False Negative
                [ld['O'], ld['B'], ld['O'], ld['O'], ld['O'], ld['B'],
                 ld['O'], ld['O'], ld['O']],  # 1 False Pos and 1 False Neg
                [ld['O'], ld['B'], ld['O'], ld['O'], ld['O'], ld['O'],
                 ld['B'], ld['I'], ld['O']],  # Correct
            ]).to(dtype=torch.int64), num_classes=3)
        md_labels = torch.Tensor(n_docs * [[
                ld['O'], ld['B'], ld['O'], ld['O'], ld['O'],
                ld['O'], ld['B'], ld['I'], ld['O']
            ]])
        ed_label_ids = n_docs * [
                [None, 'Finland', None, None, None, None,
                 'Northern_Europe', None, None]
            ]
        ed_label_vectors = torch.zeros((*input_ids.shape, 100))
        ed_label_vectors[:, 1, :] = torch.Tensor(
                kb.get_entity_vector('Finland')
            ).broadcast_to((n_docs, 100))
        ed_label_vectors[:, 6, :] = torch.Tensor(
                kb.get_entity_vector('Northern_Europe')
            ).broadcast_to((n_docs, 100))
        ed_preds = ed_label_vectors
        # Wrong vector predictions on last document
        ed_preds[4, :, :] = torch.zeros((seq_len, 100))
        # 5 different docs. Resolving overlaps tested in test_document
        docs = [1, 2, 3, 4, 5]
        positions = [(0, 7), (0, 7), (0, 7), (0, 7), (0, 7)]
        self.exp_md_tp = [2, 2, 1, 1, 2]
        self.exp_md_fp = [0, 1, 0, 1, 0]
        self.exp_md_fn = [0, 0, 1, 1, 0]
        self.exp_ed_tp = [2, 2, 1, 1, 0]
        self.exp_ed_fp = [0, 1, 0, 1, 2]
        self.exp_ed_fn = [0, 0, 1, 1, 2]

        doc_file = self.config['DATA']['Annotated Dataset']
        self.subject = Evaluator(
                md_preds=md_preds,
                md_labels=md_labels,
                ed_preds=ed_preds,
                ed_label_vectors=ed_label_vectors,
                ed_label_ids=ed_label_ids,
                docs=docs,
                positions=positions,
                knowledgebase=kb,
                documents_file=doc_file,
            )

    def test_evaluate_omniscient(self):
        """
        Check evaluation of perfect prediction (everything should be correct)
        Checks the evaluation function and md_by_doc and ed_by_doc
        Also tests that _generate_docs creates correct number of docs
        """
        vectors_dir = self.config['DATA']['Uncased Input Vectors Dir']
        dataset_generator = DatasetGenerator.load(vectors_dir)
        data_loaders = dataset_generator.split_conll_default(batch_size=2000)
        test_data = [d for d in data_loaders[2]][0]
        # print(type(test_data), len(test_data), [t.shape for t in test_data])
        md_labels = test_data[3].to(dtype=torch.float)
        md_preds = torch.nn.functional.one_hot(
                md_labels.to(dtype=torch.long),
                num_classes=4
            )[:, :, :3]
        ed_emb_labels = test_data[4].to(dtype=torch.float)
        ed_preds = ed_emb_labels.clone()
        ed_labels = dataset_generator.ed_labels[-len(ed_preds):]
        dataset_to_doc = dataset_generator.doc_indices[-len(ed_preds):]
        dataset_to_doc_pos = dataset_generator.doc_pos[-len(ed_preds):]

        kb_cg = get_knowledge_base(self.config, wiki='pedia', with_cg=True)
        doc_file = self.config['DATA']['Annotated Dataset']
        subject = Evaluator(
                md_preds=md_preds,
                md_labels=md_labels,
                ed_preds=ed_preds,
                ed_label_vectors=ed_emb_labels,
                ed_label_ids=ed_labels,
                docs=dataset_to_doc,
                positions=dataset_to_doc_pos,
                knowledgebase=kb_cg,
                documents_file=doc_file,
            )

        subject.evaluation(
                candidate_generation=True,
                use_fallback_for_empty_cs=False,
                verbose=False,
            )
        expected_docs = 231
        res_docs = len(subject.documents)
        exp_md_tp = [
            sum([1 for label in doc.labels.values()
                 if label.label_id is not None])
            for doc in subject.documents]
        exp_md = (exp_md_tp, [0] * len(subject.documents),
                  [0] * len(subject.documents))
        res_md = subject.md_by_doc()

        exp_ed_tp = []
        exp_ed_fp = []
        exp_ed_fn = []
        exp_ed_oo_kb = []
        exp_ed_no_cands = []
        # oo_kb_expected = []
        # oo_kb_result = []
        for doc in subject.documents:
            exp_ed_tp += [sum([
                    1 for label in doc.labels.values()
                    if label.in_kb is True
                    and label.in_candidate_set is True
                ])]
            exp_ed_fp += [sum([
                    1 for pred in doc.predictions.values()
                    if pred.correct_ed is False
                ])]
            exp_ed_fn += [sum([
                    1 for label in doc.labels.values()
                    if label.in_candidate_set is False
                ])]
            exp_ed_oo_kb += [sum([
                    1 for label in doc.labels.values()
                    if label.label_id is not None and
                    not kb_cg.in_kb(label.label_id)
                ])]
            exp_ed_no_cands += [sum([
                    1 for pred in doc.predictions.values()
                    if pred.no_candidates is True
                ])]
        exp_ed = (exp_ed_tp, exp_ed_fp,
                  exp_ed_fn, exp_ed_oo_kb,
                  exp_ed_fn, exp_ed_no_cands)
        res_ed = subject.ed_by_doc(verbose=False)
        self.assertEqual(expected_docs, res_docs)
        self.assertEqual(exp_md, res_md)
        self.assertEqual((1, 1, 1), subject.micro_evaluation(*res_md))
        self.assertEqual(1, subject.macro_evaluation(*res_md))
        self.assertEqual(exp_ed[0], res_ed[0])
        self.assertEqual(exp_ed[1], res_ed[1])
        self.assertEqual(exp_ed[2], res_ed[2])
        self.assertEqual(exp_ed[3], res_ed[3])
        self.assertEqual(exp_ed[4], res_ed[4])
        self.assertEqual(exp_ed[5], res_ed[5])

    def test_md_by_doc(self):
        """Check that MD evaluation is correct"""
        res = self.subject.md_by_doc()
        self.assertEqual(self.exp_md_tp, res[0])
        self.assertEqual(self.exp_md_fp, res[1])
        self.assertEqual(self.exp_md_fn, res[2])

    def test_ed_by_doc(self):
        """Check that ED evaluation is correct"""
        # ED without candidate generation
        res = self.subject.ed_by_doc(candidate_generation=False, verbose=False)
        self.assertEqual(self.exp_ed_tp, res[0])
        self.assertEqual(self.exp_ed_fp, res[1])
        self.assertEqual(self.exp_ed_fn, res[2])
        self.assertEqual(0, sum(res[3]))
        self.assertEqual(0, sum(res[4]))
        self.assertEqual(0, sum(res[5]))

    def test_macro_evaluation(self):
        """Check calculated Macro evaluation"""
        tp = self.exp_ed_tp
        fp = self.exp_ed_fp
        fn = self.exp_ed_fn
        f1s = []
        for tup in zip(tp, fp, fn):
            prec = tup[0]/(tup[0]+tup[1])
            rec = tup[0]/(tup[0]+tup[2])
            if (prec + rec) == 0:
                f1s += [0]
            else:
                f1s += [2 * prec * rec / (prec + rec)]
        exp_f1 = sum(f1s) / len(f1s)
        res = self.subject.macro_evaluation(
                self.exp_ed_tp,
                self.exp_ed_fp,
                self.exp_ed_fn
            )
        self.assertEqual(exp_f1, res)

    def test_micro_evaluation(self):
        """Check calculated Micro evaluation"""
        tp = self.exp_ed_tp
        fp = self.exp_ed_fp
        fn = self.exp_ed_fn
        exp_precision = sum(tp)/(sum(tp) + sum(fp))
        exp_recall = sum(tp)/(sum(tp) + sum(fn))
        exp_f1 = 2*(exp_precision * exp_recall) / (exp_precision + exp_recall)
        res = self.subject.micro_evaluation(
                self.exp_ed_tp,
                self.exp_ed_fp,
                self.exp_ed_fn,
            )
        self.assertEqual(exp_precision, res[0])
        self.assertEqual(exp_recall, res[1])
        self.assertEqual(exp_f1, res[2])

    def test_eval_seen_unseen(self):
        """
        Test eval_seen_unseen function with three cases:
        1. All entities are in train,
        2. Some entities are in train,
        3. no entities are in train
        """
        # Initialize documents
        self.subject.evaluation(verbose=False)

        # Case 1: All entities are in train
        train_entities_1 = [
                label.label_id
                for doc in self.subject.documents
                for label in doc.labels.values()
            ]
        exp_1 = (1, 1)
        _, res_1 = self.subject.eval_seen_unseen(train_entities_1)

        # Case 2: One of entities are in train
        train_entities_2 = [train_entities_1[0]]
        exp_2 = (1, 1)
        _, res_2 = self.subject.eval_seen_unseen(train_entities_2)

        # Case 3: None of entities are in train
        train_entities_3 = ["Fiona_Apple"]
        exp_3 = (1, 1)
        _, res_3 = self.subject.eval_seen_unseen(train_entities_3)

        self.assertEqual(exp_1, res_1)
        self.assertEqual(exp_2, res_2)
        self.assertEqual(exp_3, res_3)

"""
Functions for:
 * Accuracy calculation:
     * over mention or
     * over candidate
 * Read output file from previous evaluation, and get accuracy
 * Plot evaluation results
"""

import time
import torch
from typing import Tuple, List, Optional
from collections import Counter
from copy import deepcopy
from src.toolbox import get_docs
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia
from src.document import get_document_from_sequences

import random
import os
# Fixed random seed to always sample same subset of documents for evaluation
random.seed(12)


class Evaluator:
    def __init__(
                self,
                md_preds: torch.Tensor,
                md_labels: torch.Tensor,
                ed_preds: torch.Tensor,
                ed_label_vectors: torch.Tensor,
                ed_label_ids: List,
                docs: List,
                positions: List,
                knowledgebase: KnowledgeBaseWikipedia,
                documents_file: str = '',
            ):
        """
        :param md_preds: Tensor of predicted mention detection logits
            in shape N * sequence length * 3
        :param md_labels: Tensor of IOB labels as values 0 ('I'), 1 ('O'),
            2 ('B') or 3 ('None'), in shape N * sequence length
        :param ed_preds: Tensor of entity embeddings from the model
            with shape N * sequence length * embedding length
        :param ed_labels_vectors: a tensor of ground truth ED vectors
        :param ed_labels_ids: List of label ID for all tokens
            ('None' where no label), with shape N * sequence length
        :param docs: a list of doc IDs for each data sequence
        :param positions: the positions of each of the sequences
            in their origin documents
        :param knowledgebase: A KnowledgeBase object
        :param documents_file: file with annotated documents
            (necessary for candidate generation)
        """
        if not len(md_preds) == len(md_labels):
            raise Exception(
                    f"MD predictions and labels must be same length. "
                    f"Got lengths: {len(md_preds)}, {len(md_labels)}"
                )
        if not len(ed_preds) == len(ed_label_ids):
            raise Exception(
                    f"ED predictions and labels must be same length. "
                    f"Got lengths: {len(ed_preds)}, {len(ed_label_ids)}"
                )
        if not len(ed_preds) == len(md_preds):
            raise Exception(
                    f"ED and MD predictions and labels must be same length. "
                    f"Got lengths: ED {len(ed_preds)}, and MD {len(md_preds)}"
                )

        # Immediately resolve overlaps between sequences from same document,
        #  and remove padding
        self.documents = self._generate_docs(
                md_preds,
                md_labels,
                ed_preds,
                ed_label_vectors,
                ed_label_ids,
                docs,
                positions,
                doc_file=documents_file,
            )

        self.knowledgebase = knowledgebase
        # Has the evaluate function been run ?
        self.evaluated = False

    def evaluation(
                self,
                sample_n: Optional[int] = 0,
                candidate_generation: bool = True,
                use_fallback_for_empty_cs: bool = True,
                verbose: bool = True,
            ):
        """
        Calculate and return precision, recall, and micro and macro F1-score
            of the combined MD and ED predictions

        :param sample_n: Number of documents to randomly sample for evaluation
            if < 0, all documents are evaluated
        :param candidate_generation: if True, using candidate generation
            for evaluation
        :param use_fallback_for_empty_cs: if True,
            and using candidate generation, using fallback to brute-force
            search over all Wikipedia2vec vectors. This is slower,
            and not recommended during training.
        :param verbose: if True, print progress
        :returns: a tuple of in-KB micro FD1 score for MD and ED
        """
        # Get sampled subset of sample_n documents
        if sample_n > 0 and sample_n < len(self.documents):
            indices = random.sample(range(len(self.documents)), sample_n)
            self.documents = [self.documents[i] for i in indices]
        # Get lists of True Positive, False Positive, and False Negative
        #   MD spans by document
        md_tp, md_fp, md_fn = self.md_by_doc()

        # Get Entity Disambiguation metrics:
        ed_tp, ed_fp, ed_fn, _, _, _ = \
            self.ed_by_doc(
                    candidate_generation,
                    use_fallback_for_empty_cs,
                    verbose
                )

        _, _, in_kb_md_micro_f1 = \
            self.micro_evaluation(md_tp, md_fp, md_fn)

        # InKB evaluation:
        # Micro: precision, recall and F1 across all mentions in all docs
        _, _, in_kb_ed_micro_f1 = \
            self.micro_evaluation(ed_tp, ed_fp, ed_fn)

        self.evaluated = True

        return in_kb_md_micro_f1, in_kb_ed_micro_f1

    def md_by_doc(self):
        """
        Get doc-wise True Positive, False Positive and False Negative
        :returns: three lists of number of true positive,
            false positive and false negative by document
        """
        md_tp = []
        md_fp = []
        md_fn = []
        for document in self.documents:
            doc_tp, doc_fp, doc_fn = document.get_md_stats()
            md_tp += [doc_tp]
            md_fp += [doc_fp]
            md_fn += [doc_fn]
        return md_tp, md_fp, md_fn

    def ed_by_doc(
                self,
                candidate_generation: bool = True,
                use_fallback_for_empty_cs: bool = True,
                verbose: bool = True,
            ):
        """
        Method to sort ED label and prediction into
            doc-wise True Positive, False Positive and False Negative lists
        Takes output of get_sorted_md_spans as input

         * A predicted span is a ...
           * True Positive if predicted span matches the label span
                (a span starting with 'I' is treated as a valid span),
                AND predicted ED entity matches label entity
           * False Positive if a predicted span is not a label span,
                OR the span was predicted but the predicted ED entity was wrong
         * A label span is a False Negative if the label span was not predicted

        :param candidate_generation: if provided, triggers candidate sets
            in knowledge base. Entities that are correctly prediced by MD,
            but are not in the KB's candidate gets in_candidate_set = False
        :param use_fallback_for_empty_cs: if True,
            and using candidate generation, using fallback to brute-force
            search over all Wikipedia2vec vectors. This is slower,
            and not recommended during training.
        :param verbose: if True, print progress of ED evaluation
        :returns: four lists of lists by doc of true positive,
            false positive and false negative spans,
            and the spans that are not in the knowledgebase
            Each span is a tuple of span start and end in the document
        """
        ed_tp = []
        ed_fp = []
        ed_fn = []
        out_of_kb = []
        not_in_cands = []
        no_cands = []

        if verbose:
            n_docs = len(self.documents)
            print()
            print(f"ED evaluation: doc {0:>4} / {n_docs}", end='\r')

        # Iterate documents
        for i_doc, doc in enumerate(self.documents):
            if not self.evaluated:
                # Start the ED evaluation on each document
                doc.ed_prediction(
                        self.knowledgebase,
                        cand_gen=candidate_generation,
                        use_fallback_for_empty_cs=use_fallback_for_empty_cs,
                    )
            # Get the result of the evaluation
            doc_tp, doc_fp, doc_fn, doc_oo_kb, doc_oo_cand, doc_no_cand = \
                doc.get_ed_stats()
            ed_tp += [doc_tp]
            ed_fp += [doc_fp]
            ed_fn += [doc_fn]
            out_of_kb += [doc_oo_kb]
            not_in_cands += [doc_oo_cand]
            no_cands += [doc_no_cand]
            # Find performance so far to show a meaningful progress
            eval_res = self.micro_evaluation(ed_tp, ed_fp, ed_fn)
            if verbose:
                print(
                        f"ED evaluation, doc {i_doc+1:>4} / {n_docs},"
                        f" current result: "
                        f"prec: {eval_res[0]*100:5.2f} %, "
                        f"rec: {eval_res[1]*100:5.2f} %, "
                        f"micro F1: {eval_res[2]*100:5.2f}",
                        end='\r'
                    )
        if verbose:
            print()
        self.evaluated = True
        return ed_tp, ed_fp, ed_fn, out_of_kb, not_in_cands, no_cands

    def dump_pred_to_file(
                self,
                out_dir: str,
                file_prefix: str = "",
            ):
        """
        Print ground truth and model predictions to a .tsv file in model_dir
        for evaluation by the "score_aida.py" script from Poerner 2019 (E-BERT)
        https://github.com/NPoe/ebert/blob/
333b789593df1aebb967e8d298197ebdaeea032e/code/score_aida.py
        :param out_dir: path to output dir for resulting file
        :param file_prefix: e.g. "conll_test" for test set
        """
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        # In-KB evaluation
        f_pred_in_kb = open(os.path.join(
                    out_dir,
                    file_prefix + "_in_kb_predictions.tsv"
                ), 'w')
        f_gt_in_kb = open(os.path.join(
                    out_dir,
                    file_prefix + "_in_kb_ground_truth.tsv"
                ), 'w')
        # Including Out-of-KB evaluation
        f_pred_oo_kb = open(os.path.join(
                    out_dir,
                    file_prefix + "_oo_kb_predictions.tsv"
                ), 'w')
        f_gt_oo_kb = open(os.path.join(
                    out_dir,
                    file_prefix + "_oo_kb_ground_truth.tsv"
                ), 'w')
        f_str = "{}\t{}\t{}\t{}\t{}\n"
        # Start position for current document in the chain of documents
        start_pos = 0
        for doc in self.documents:
            for pred in doc.predictions.values():
                # Ignore predictions on un-IDed mentions
                if pred.span in doc.labels \
                        and doc.labels[pred.span].label_id is None:
                    continue

                # # For token position in "all documents as one"
                # #   used for evaluation in ELEVANT
                # str_pred = f_str.format(
                #         start_pos + pred.span[0],
                #         start_pos + pred.span[1],
                #         pred.predicted_id,
                #         pred.mention_text,
                #         ';'.join(pred.candidates),
                #     )

                # For doc_id, character positions and candidates
                str_pred = f_str.format(
                        doc.doc_id,
                        sum(len(token)
                            for token in doc.doc_text[:pred.span[0]]
                            ) + pred.span[0],
                        len(' '.join(doc.doc_text[:1+pred.span[1]])) - 1,
                        pred.predicted_id,
                        ';'.join(pred.candidates),
                    )
                # Write to in_kb if in KB
                if pred.in_kb is not False:
                    f_pred_in_kb.write(str_pred)
                # Write to oo_kb file either way
                f_pred_oo_kb.write(str_pred)
            for label in doc.labels.values():
                # Ignore un-IDed mentions
                if label.label_id is None:
                    continue
                # # Used for evaluation in ELEVANT
                # str_gt = f_str.format(
                #         start_pos + label.span[0],
                #         start_pos + label.span[1],
                #         label.label_id,
                #         label.mention_text,
                #     )

                # For doc_id and character positions
                str_gt = f_str.format(
                        doc.doc_id,
                        sum(len(token)
                            for token in doc.doc_text[:label.span[0]]
                            ) + label.span[0],
                        len(' '.join(doc.doc_text[:1+label.span[1]])) - 1,
                        label.label_id,
                        label.mention_text,
                    )

                if label.in_kb is True:
                    f_gt_in_kb.write(str_gt)
                # Write to oo_kb file either way
                f_gt_oo_kb.write(str_gt)
            start_pos += doc.doc_len
        f_pred_in_kb.close()
        f_pred_oo_kb.close()
        f_gt_in_kb.close()
        f_gt_oo_kb.close()

    def print_documents(self):
        """Prints formatted documents with predictions and labels in colour"""
        for doc in self.documents:
            print(f"Doc {doc.doc_id}:")
            print(doc)

    def evaluation_str(self):
        """
        Prints the evaluation metrics found by the evaluation method.
        Requires that the evaluation method has run already
        """
        if not self.evaluated:
            raise Exception(
                    "Evaluation metrics are missing. "
                    "Run evaluation method before calling print method."
                )

        #    1: Collect all the metrics
        md_tp, md_fp, md_fn = self.md_by_doc()
        # Micro: precision, recall and F1 across all mentions in all docs
        md_precision, md_recall, md_micro_f1 = \
            self.micro_evaluation(md_tp, md_fp, md_fn)
        # Macro: precision, recall and F1 averaged by document
        md_macro_f1 = self.macro_evaluation(md_tp, md_fp, md_fn)

        # ED evaluation
        ed_tp, ed_fp, ed_fn, oo_kb, oo_cand, no_cands = self.ed_by_doc()

        # InKB evaluation:
        # Micro: precision, recall and F1 across all mentions in all docs
        in_kb_ed_precision, in_kb_ed_recall, in_kb_ed_micro_f1 = \
            self.micro_evaluation(ed_tp, ed_fp, ed_fn)
        # Macro: averaged F1 score by document
        in_kb_ed_macro_f1 = self.macro_evaluation(ed_tp, ed_fp, ed_fn)

        # Out-of-KB evaluation (punishes limited KBs):
        # Add the out-of-KB spans as if they were wrong predictions
        oo_kb_ed_fp = []
        oo_kb_ed_fn = []
        for i in range(len(oo_kb)):
            oo_kb_ed_fp += [ed_fp[i] + oo_kb[i]]
            oo_kb_ed_fn += [ed_fn[i] + oo_kb[i]]
        # Recalculate micro and macro with the Out-of-KB errors
        oo_kb_ed_precision, oo_kb_ed_recall, oo_kb_ed_micro_f1 = \
            self.micro_evaluation(ed_tp, oo_kb_ed_fp, oo_kb_ed_fn)
        # Macro: averaged F1 score by document
        oo_kb_ed_macro_f1 = \
            self.macro_evaluation(ed_tp, oo_kb_ed_fp, oo_kb_ed_fn)

        n_md_tp = sum(md_tp)
        n_md_fn = sum(md_fn)
        n_md_fp = sum(md_fp)
        n_ed_tp = sum(ed_tp)
        n_ed_fn = sum(ed_fn)
        n_ed_fp = sum(ed_fp)
        n_oo_kb = sum(oo_kb)
        n_oo_kb_ed_fn = sum(oo_kb_ed_fn)
        n_oo_kb_ed_fp = sum(oo_kb_ed_fp)
        # Errors only due to wrong disambiguation
        # In-KB MD errors
        n_md_fn_in_kb = 0
        n_md_tp_in_kb = 0
        for doc in self.documents:
            for label in doc.labels.values():
                if label.in_kb:
                    if label.predicted_md is False:
                        n_md_fn_in_kb += 1
                    elif label.predicted_md is True:
                        n_md_tp_in_kb += 1
        n_ed_error = n_ed_fn - n_md_fn

        # Finding the number of unique entities
        ent_md_tp = []
        ent_md_fn = []
        ent_in_kb = []
        ent_oo_kb = []
        ent_ed_tp = []
        ent_ed_fn_oo_kb = []
        ent_ed_fn_in_kb = []
        ent_cs_fn = []
        ent_not_in_cs = []
        ent_empty_cs = []

        def add_to_list(id, value, list_true, list_false):
            # If value is True, add id to list_true, if False add to list_false
            if value is True:
                list_true += [id]
            elif value is False:
                list_false += [id]

        # To find number of unique entities in each category
        for doc in self.documents:
            for label in doc.labels.values():
                id_ = label.label_id
                if id_ is None:
                    continue
                add_to_list(id_, label.predicted_md, ent_md_tp, ent_md_fn)
                add_to_list(id_, label.in_kb, ent_in_kb, ent_oo_kb)
                pred_ed = label.predicted_ed
                add_to_list(id_, pred_ed, ent_ed_tp, ent_ed_fn_oo_kb)
                # Conditionally add to in_kb_fn and in_cs_fn lists
                #   (ed_tp is always the same)
                if label.in_kb is True:
                    add_to_list(id_, pred_ed, [], ent_ed_fn_in_kb)
                if label.in_candidate_set is True:
                    add_to_list(id_, pred_ed, [], ent_cs_fn)
                elif label.in_candidate_set is False:
                    ent_not_in_cs += [id_]
                    # If the candidate set was empty
                    if doc.predictions[label.span].no_candidates is True:
                        ent_empty_cs += [id_]

        n_unique_md_tp = len(set(ent_md_tp))
        n_unique_md_fn = len(set(ent_md_fn))
        n_unique_in_kb = len(set(ent_in_kb))
        n_unique_oo_kb = len(set(ent_oo_kb))
        n_unique_ed_tp = len(set(ent_ed_tp))
        n_unique_ed_fn_oo_kb = len(set(ent_ed_fn_in_kb))
        n_unique_ed_fn_in_kb = len(set(ent_ed_fn_oo_kb))
        n_unique_cs_fn = len(set(ent_cs_fn))
        n_unique_empty_cs = len(set(ent_empty_cs))
        n_unique_not_in_cs = len(set(ent_not_in_cs))

        #    2: Format result strings and print them
        # ANSI colors
        c = {
                'e': "\033[0;0m",  # End
                'g': "\033[92m",  # Green
                'r': "\033[91m",  # Red
            }
        # Build the output string
        f_str = """Evaluation:
- Mention Detection (MD) (strong match):
  {:>6}   Total spans (mentions with an ID)
  {}{:>6}{}    True Positives (correct span found)
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   False Negatives (label span not found)
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   False Positives (predicted span is wrong)
  {:6.2f} % MD Precision
  {:6.2f} % MD Recall
  {:6.2f}   MD Micro F1 Score (across dataset)
  {:6.2f}   MD Macro F1 Score (avg. F1 by document)"""
        f_str = f_str.format(
                n_md_tp + n_md_fn,  # Total spans
                c['g'], n_md_tp, c['e'],  # MD True Positive
                c['g'], n_unique_md_tp, c['e'],  # MD True Positive
                c['r'], n_md_fn, c['e'],  # MD False Negative
                c['r'], n_unique_md_fn, c['e'],  # MD True Positive
                c['r'], n_md_fp, c['e'],  # MD False Positive
                100 * md_precision,  # MD Precision
                100 * md_recall,  # MD Recall
                100 * md_micro_f1,  # MD Micro F1 Score
                100 * md_macro_f1,  # MD Macro F1 Score
            )
        f_str += """\n- Entity Linking (EL):
-- Knowledge Base (KB)
  {:>6}   Spans with label (mentions with an ID)
  {}{:>6}{}   mentions with ID in KB
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   mentions with ID not in KB
 ({}{:>6}{}   ... of which unique entities )
-- Out-of-KB EL (ID not in KB treated as an error):
  {}{:>6}{}   True Positive (correct entity)
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   False Negative = {} not in KB + {} MD FN + {} Wrong ED
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   False Positive = {} not in KB + {} MD FP + {} Wrong ED
  {:6.2f} % OO-KB EL Precision
  {:6.2f} % OO-KB EL Recall
  {:6.2f}   OO-KB EL Micro F1 Score (across dataset)
  {:6.2f}   OO-KB EL Macro F1 Score (avg. F1 by document)"""
        f_str = f_str.format(
                n_ed_tp + n_ed_fn + n_oo_kb,  # Spans with label
                c['g'], n_ed_tp + n_ed_fn, c['e'],  # in KB
                c['g'], n_unique_in_kb, c['e'],  # in KB entities
                c['r'], n_oo_kb, c['e'],  # OO KB
                c['r'], n_unique_oo_kb, c['e'],  # OO KB entities
                c['g'], n_ed_tp, c['e'],  # OO KB True Positive
                c['g'], n_unique_ed_tp, c['e'],  # OO KB TP entities
                c['r'], n_oo_kb_ed_fn, c['e'],  # OO KB False Negative
                n_oo_kb, n_md_fn_in_kb, n_ed_error,
                c['r'], n_unique_ed_fn_oo_kb, c['e'],  # OO KB FN entities
                c['r'], n_oo_kb_ed_fp, c['e'],  # OO KB False Positive
                n_oo_kb, n_md_fp, n_ed_error,
                100 * oo_kb_ed_precision,  # OO KB ED Precision
                100 * oo_kb_ed_recall,  # OO KB ED Recall
                100 * oo_kb_ed_micro_f1,  # OO KB ED Micro F1 Score
                100 * oo_kb_ed_macro_f1,  # OO KB ED Macro F1 Score
            )

        f_str += """\n-- In-KB EL (performance on entities in KB):
  {}{:>6}{}   True Positive (correct entity)
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   False Negative = {} MD FN + {} Wrong ED
 ({}{:>6}{}   ... of which unique entities )
  {}{:>6}{}   False Positive = {} MD FP + {} Wrong ED
  {:6.2f} % In-KB EL Precision
  {:6.2f} % In-KB EL Recall
  {:6.2f}   In-KB EL Micro F1 Score (across dataset)
  {:6.2f}   In-KB EL Macro F1 Score (avg. F1 by document)"""

        f_str = f_str.format(
                c['g'], n_ed_tp, c['e'],  # In-KB True Positive
                c['g'], n_unique_ed_tp, c['e'],  # In-KB True Positive
                c['r'], n_ed_fn, c['e'],  # In-KB False Negative
                n_md_fn_in_kb, n_ed_error,
                c['r'], n_unique_ed_fn_in_kb, c['e'],  # In-KB FN entities
                c['r'], n_ed_fp, c['e'],  # In-KB False Positive
                n_md_fp, n_ed_error,
                100 * in_kb_ed_precision,  # In-KB ED Precision
                100 * in_kb_ed_recall,  # In-KB ED Recall
                100 * in_kb_ed_micro_f1,  # In-KB ED Micro F1 Score
                100 * in_kb_ed_macro_f1,  # In-KB ED Macro F1 Score
            )

        if self.knowledgebase.cand_gen:
            # Accuracy of ED predictions of entities that are in KB and CS
            #  is (TP) / (TP + (FP from ED) - (FP from entity not in CSs))
            n_oo_cand = sum(oo_cand)
            in_cs_accuracy = n_ed_tp \
                / (n_ed_tp + n_ed_error - n_oo_cand)

            f_str += """\n- Entity Disambiguation (ED) (In-CS EL):
  {}{:>6}{}   Mentions with label not in CS
 ({}{:>6}{}   ... of which unique entities)
  {}{:>6}{}   Mentions for which CS was empty
 ({}{:>6}{}   ... of which unique entities)
  {:>6}   Correct mention found, and label is in CS
  {}{:>6}{}   Predicted entity correct
 ({}{:>6}{}   ... of which unique entities)
  {}{:>6}{}   Predicted entity wrong (correct candidate in CS, but chose wrong)
 ({}{:>6}{}   ... of which unique entities)
  {:6.2f} % ED Accuracy"""
            f_str = f_str.format(
                    c['r'], n_oo_cand, c['e'],  # Not in candidate set
                    c['r'], n_unique_not_in_cs,
                    c['e'],  # Entities not in candidate set
                    c['r'], sum(no_cands), c['e'],  # Candidate set empty
                    c['r'], n_unique_empty_cs,
                    c['e'],  # Entities not in candidate set
                    n_md_tp_in_kb - n_oo_cand,  # In KB and CS
                    c['g'], n_ed_tp, c['e'],  # In KB and CS and correct
                    c['g'], n_unique_ed_tp, c['e'],  # Entities correct
                    c['r'], n_ed_error - n_oo_cand,
                    c['e'],  # In KB and CS but wrong
                    c['r'], n_unique_cs_fn, c['e'],  # In wrong entities
                    100 * in_cs_accuracy,  # ED accuracy
                )

        return f_str

    @classmethod
    def macro_evaluation(cls, tp, fp, fn):
        """Get macro F1 score (average of the micro F1 of each document)"""
        doc_f1s = []
        for doc_tp, doc_fp, doc_fn in zip(tp, fp, fn):
            _, _, doc_f1 = cls.micro_evaluation([doc_tp], [doc_fp], [doc_fn])
            doc_f1s += [doc_f1]
        md_macro_f1 = sum(doc_f1s) / len(doc_f1s)
        return md_macro_f1

    @classmethod
    def micro_evaluation(cls, tp, fp, fn):
        """Get precision recall and macro F1 score (averaged over mentions)"""
        # Corner case: Empty document (for single document from macro)
        if sum([sum(tp), sum(fp), sum(fn)]) == 0:
            return 1, 1, 1
        if sum(tp) + sum(fp) == 0:
            md_precision = 0
        else:
            md_precision = sum(tp) / (sum(tp) + sum(fp))
        if sum(tp) + sum(fn) == 0:
            md_recall = 0
        else:
            md_recall = sum(tp) / (sum(tp) + sum(fn))

        md_micro_f1_nominator = 2 * (md_precision * md_recall)
        md_micro_f1_denominator = (md_precision + md_recall)
        if md_micro_f1_denominator == 0:
            md_micro_f1 = 0
        else:
            md_micro_f1 = md_micro_f1_nominator / (md_precision + md_recall)

        return md_precision, md_recall, md_micro_f1

    @classmethod
    def _generate_docs(
                cls,
                md_preds: torch.Tensor,
                md_labels: torch.Tensor,
                ed_preds: torch.Tensor,
                ed_label_vectors: torch.Tensor,
                ed_label_ids: List,
                docs: List[int],
                positions: List[Tuple[int, int]],
                doc_file: Optional[str] = None,
            ) -> Tuple:
        """
        :param md_preds: MD predictions
        :param md_labels: MD labels
        :param ed_preds: ED predictions
        :param ed_label_vectors: ED ground truth vectors
        :param ed_label_ids: ED ground truth entitiy IDs
        :param docs: List of document index for each tensor along dimension 0
        :param positions: list of positions in origin document for each tensor
        :param doc_file: path to a documents file. Necessary for ED with
            candidate sets, to get the correct mention text
        :returns: a list of Document objects with resolved overlaps
        """
        documents = []
        # Count sequences for each doc
        doc_counts = Counter(docs)
        # If doc_file is provided, get document generator
        if doc_file:
            # Document generator
            docs_list = list(get_docs(doc_file))
        else:
            docs_list = None

        # Start at doc at index 0
        i_docs = 0
        # Iterate batches of sequences by document
        while i_docs < len(md_preds):
            # Current doc ID
            doc_id = docs[i_docs]
            # Current document tokens, if relevant
            if docs_list:
                doc_ = docs_list[doc_id]
                doc_tokens = doc_.tokens
                doc_text = [token.text for token in doc_tokens]
            else:
                doc_text = []
            # Number of sequences from current doc
            n_data_points = doc_counts[doc_id]
            # All the position tuples of sequences from the current doc
            positions_doc = positions[i_docs:i_docs+n_data_points]
            documents += [get_document_from_sequences(
                    doc_id,
                    positions_doc,
                    md_preds[i_docs:i_docs+n_data_points],
                    md_labels[i_docs:i_docs+n_data_points],
                    ed_preds[i_docs:i_docs+n_data_points],
                    ed_label_vectors[i_docs:i_docs+n_data_points],
                    ed_label_ids[i_docs:i_docs+n_data_points],
                    doc_text,
                )]
            i_docs += n_data_points

        return documents

    @classmethod
    def format_eval_output(
                cls,
                duration: float,
                md_acc: float,
                ed_acc: float,
                avg_loss: float
            ):
        """
        Format a summary string of the evaluation (used between epochs)
        :param duration: duration of evaluation
        :param md_acc: MD accuracy
        :param ed_acc: ED accuracy
        :param avg_loss: Average loss
        """
        duration = time.strftime("%H:%M:%S", time.gmtime(duration))
        format_str = f"       Duration: {duration}" \
                     f"\n  Avgerage loss: {avg_loss:.4f}" \
                     f"\n    MD accuracy: {md_acc:.4f}" \
                     f"\n    ED accuracy: {ed_acc:.4f}"
        return format_str

    def eval_seen_unseen(
                self,
                train_entities: list,
            ):
        """
        Evaluate the performance of the model on the current dataset,
            grouped by entities in training dataset or not
        :param train_entities: List of entities (by ID) from the training set
        """
        if not self.evaluated:
            raise Exception(
                    "Evaluation metrics are missing. "
                    "Run evaluation method before calling print method."
                )

        # Make sure the train entities is a set and without None
        if type(train_entities[0]) is list:
            # Flaten a list-in-list
            train_entities = [entity for doc_entities in train_entities
                              for entity in doc_entities]
        for i_ent, ent in enumerate(train_entities):
            if ent is not None:
                train_entities[i_ent] = ent.replace('_', ' ')
        # Count occurrences of each label
        train_entity_occurrences = Counter(train_entities)
        if None in train_entity_occurrences:
            train_entity_occurrences.pop(None)

        # Hold ED stats for in-train entities
        ed_tp_in_train = []
        ed_fp_in_train = []
        ed_fn_in_train = []

        # Hold ED stats for not-in-train entities
        ed_tp_not_in_train = []
        ed_fp_not_in_train = []
        ed_fn_not_in_train = []

        # Entities in train that the model gets right
        tp_not_in_train = []
        # Entities not in train that the model gets wrong
        fn_not_in_train = []
        # Entities in train that the model gets right
        tp_in_train = []
        # Entities in train that the model gets wrong
        fn_in_train = []

        # First, make new documents with entities in train and not in train
        for doc in self.documents:
            # Make copy documents
            doc_in_train = deepcopy(doc)
            doc_not_in_train = deepcopy(doc)

            # Iterate labels
            for label_span, label in doc.labels.items():
                # If entity is not in KB, MD not predicted,
                #   entity not in CS, or entity label is None
                if label.label_id is None \
                        or label.predicted_md is False \
                        or label.in_candidate_set is False \
                        or label.in_kb is False:
                    # Remove from both documents
                    doc_not_in_train.labels.pop(label_span)
                    doc_in_train.labels.pop(label_span)
                    # Remove corresponding prediction
                    if label_span in doc.predictions:
                        doc_not_in_train.predictions.pop(label_span)
                        doc_in_train.predictions.pop(label_span)
                    continue

                # If this entity is in training set
                if label.label_id in train_entity_occurrences:
                    # Remove from not_in_train document
                    doc_not_in_train.labels.pop(label_span)
                # If this entity is not in training set
                else:
                    # Remove from in_train document
                    doc_in_train.labels.pop(label_span)

            # Iterate predictions
            for pred_span, pred in doc.predictions.items():
                # Skip the predictions already removed
                if pred_span not in doc_in_train.predictions:
                    continue
                # If prediction has correct MD
                if pred.correct_md:
                    # Current prediction's label ID
                    pred_label_id = doc.labels[pred_span].label_id
                    # If the corresponding label is in train
                    if pred_label_id in train_entity_occurrences:
                        # Remove from not in train
                        doc_not_in_train.predictions.pop(pred_span)
                    # If the label is not in train
                    else:
                        # Remove from in train
                        doc_in_train.predictions.pop(pred_span)
                # If incorrect MD, we're not interested in it
                else:
                    # Remove from both lists
                    doc_not_in_train.predictions.pop(pred_span)
                    doc_in_train.predictions.pop(pred_span)

            # Next, run evaluation on the two new document objects
            # In-train stats
            in_train_ed_res = doc_in_train.get_ed_stats()
            ed_tp_in_train += [in_train_ed_res[0]]
            ed_fp_in_train += [in_train_ed_res[1]]
            ed_fn_in_train += [in_train_ed_res[2]]

            # Iterate the False Negatives that are in train,
            #  and add to list of TP or FN
            for label in doc_in_train.labels.values():
                if label.predicted_ed is False:
                    fn_in_train += [label.label_id]
                elif label.predicted_ed is True:
                    tp_in_train += [label.label_id]

            # Not-in-train stats
            not_in_train_ed_res = doc_not_in_train.get_ed_stats()
            ed_tp_not_in_train += [not_in_train_ed_res[0]]
            ed_fp_not_in_train += [not_in_train_ed_res[1]]
            ed_fn_not_in_train += [not_in_train_ed_res[2]]

            # Iterate the False Negatives that are in train,
            #  and add to list of TP or FN
            for label in doc_not_in_train.labels.values():
                if label.predicted_ed is False:
                    fn_not_in_train += [label.label_id]
                if label.predicted_ed is True:
                    tp_not_in_train += [label.label_id]

        # Make a summary:
        # In-train accuracy
        in_train_acc, _, _ = \
            self.micro_evaluation(
                ed_tp_in_train, ed_fp_in_train, ed_fn_in_train
            )
        # Not-in-train accuracy
        not_in_train_acc, _, _ = \
            self.micro_evaluation(
                ed_tp_not_in_train, ed_fp_not_in_train, ed_fn_not_in_train
            )

        def in_train_stats(entity_list):
            if len(entity_list) == 0:
                return (0,)*6
            # Convert to Counter dictionaries and get stats for this dataset
            count = Counter(entity_list)
            dataset_avg = sum(count.values())/len(count)
            # Get count of occurrences in train
            train_counts = [train_entity_occurrences[ent] for ent in count]
            tr_avg = sum(train_counts)/len(count)
            tr_min = min(train_counts)
            tr_min_n = Counter(train_counts)[tr_min]
            tr_max = max(train_counts)
            tr_max_n = Counter(train_counts)[tr_max]
            return tr_avg, tr_min, tr_min_n, tr_max, tr_max_n, dataset_avg

        def not_in_train_avg(entity_list):
            if len(entity_list) == 0:
                return 0
            count = Counter(entity_list)
            return sum(count.values())/len(count)

        f_str = """Evaluation of ED results sorted by entities in the
training set, and not in the training set:
- In-training-set evaluation:
  \033[92m{:>6}\033[0;0m   True Positive (correct entity)
  \033[91m{:>6}\033[0;0m   False Negative (wrong label predicted)
  {:6.2f} % Accuracy
-- \033[92mCorrect ED\033[0;0m:
  {:>6.2f}   Average occurrences in training data
  {:>6}   Minimum occurrences in train ({} entities)
  {:>6}   Maximum occurrences in train ({} entities)
  {:>6.2f}   Average occurrences in this dataset
-- \033[91mWrong ED\033[0;0m:
  {:>6.2f}   Average occurrences in training data
  {:>6}   Minimum occurrences in train ({} entities)
  {:>6}   Maximum occurrences in train ({} entities)
  {:>6.2f}   Average occurrences in this dataset
- Not-in-training-set evaluation:
  \033[92m{:>6}\033[0;0m   True Positive (correct entity)
  \033[91m{:>6}\033[0;0m   False Negative (wrong label predicted)
  {:6.2f} % Accuracy
-- \033[92mCorrect ED\033[0;0m:
  {:>6.2f}   Average occurrences in this dataset
-- \033[91mWrong ED\033[0;0m:
  {:>6.2f}   Average occurrences in this dataset"""
        f_str = f_str.format(
                sum(ed_tp_in_train),
                sum(ed_fn_in_train),
                100 * in_train_acc,
                *in_train_stats(tp_in_train),
                *in_train_stats(fn_in_train),
                sum(ed_tp_not_in_train),
                sum(ed_fn_not_in_train),
                100 * not_in_train_acc,
                not_in_train_avg(tp_not_in_train),
                not_in_train_avg(fn_not_in_train),
            )

        return f_str, (in_train_acc, not_in_train_acc)

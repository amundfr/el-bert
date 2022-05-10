"""
Classes to handle documents, including label entities and predictions.

Note: Document tokens are defined as WordPiece tokens, NOT whole words.
The Document class takes care of conversion from WorkPieces to words,
and resolves conflicts of overlapping sequences from the same document
"""

from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia
from src.input_data_generator import InputDataGenerator
from typing import Tuple, List
from torch import Tensor, cat


# Get dictionary of IOB labels from InputDataGenerator
IOB_LABEL = InputDataGenerator.IOB_LABEL


class EntityLabel:
    def __init__(
                self,
                span: Tuple[int],
                label_id: str,
                vector: Tensor,
                mention_text: str,
            ):
        """
        Stores a ground truth document entity.
        :param span: tuple of start and end position of the entity
            in the document text tokens (Document.doc_text)
        :param label_id: Ground truth QID of the entity
        :param vector: Embedding of the entity from the KB
        :param mention_text: The mention text of a label entity
        """
        # Position in Wordpiece Input IDs sequence
        self.span = span
        # Label QID
        self.label_id = label_id
        # Entity vector from KB
        self.vector = vector
        # Original mention text from origin document for the GT span
        self.mention_text = mention_text

        # Does the GT entity have a correct prediction ?
        self.predicted_md = None
        # Is the entity in the recently used knowledge base ?
        self.in_kb = None
        # Was the correct entity in the candidate set for this mention ?
        #   (only initialized if predicted by MD)
        self.in_candidate_set = None
        # Was the entity correctly classified by a prediction
        #   with the recently used knowledge base ?
        self.predicted_ed = None


class EntityPrediction:
    def __init__(
                self,
                span: Tuple[int],
                md_pred: Tensor,
                ed_embedding_pred: Tensor,
                mention_text: str,
            ):
        """
        Stores a prediction of a span, and the predicted embedding
            and the predicted embedding of the first token
        :param span: tuple of start and end position of the predicted entity
            in the document text tokens (Document.doc_text)
        :param md_pred: Predicted MD label (as returned from model)
        :param ed_embedding_pred: Predicted embedding of the entity
        :param mention_text: The mention text of a predicted entity
        """
        self.span = span
        self.md_pred = md_pred
        self.ed_embedding_pred = ed_embedding_pred
        # Original mention text from origin document for the predicted span
        self.mention_text = mention_text
        # Placeholder for predicted ID from a knowledge base
        self.predicted_id = None
        # Holds the candidates for this mention
        self.candidates = []

        # If there were no candidates found for this entity
        #   from candidate generation
        self.no_candidates = None
        # True if there is a EntityLabel with a QID and the predicted span,
        # False if there is no EntityLabel with the predicted span,
        # None if there is an EntityLabel with no QID and the predicted span,
        self.correct_md = None

        # True if MD is correct, and the evaluation KB gives label QID when
        #   given the ed_embedding_pred
        # False if:
        #   MD is incorrect,
        #   evaluation KB returned no candidates for the embedding,
        #   or evaluation KB shows wrong predicted entity
        # None if:
        #   MD is correct, but the predicted label has no QID label,
        #   or label entity is not in evaluation KB
        self.correct_ed = None
        # False if correct_md = True, but the label was not in KB
        self.in_kb = None


class Document:
    def __init__(
                self,
                doc_id: int,
                doc_text: List[str],
                md_predictions: Tensor,
                md_labels: Tensor,
                ed_embedding_predictions: Tensor,
                ed_label_vectors: Tensor,
                ed_label_ids: list,
            ):
        """
        Holds the info about a document necessary for evaluation.

        Upon initialization, resolves overlaps
            and resolves the competing predictions
            using the magnitude of MD predictions.

        :param doc_id: Document ID as integer
        :param doc_text: List of the original word tokens for this document.
            Necessary to get the right mention text for candidate generation.
        :param md_predictions:
            Tensor of logit Mention Detection predictions for each token
        :param md_labels: List of Mention Detection IOB-labels for each token
        :param ed_embedding_predictions: 
            Tensor of predicted Entity Disambiguation embeddings
        :param ed_label_vectors:
            List of ground truth entity vectors for each token
        :param ed_label_ids:
            List of ground truth entity IDs for each token
        """
        seq_len = len(md_labels)
        if not (
                    seq_len == len(md_predictions)
                    and seq_len == len(ed_embedding_predictions)
                    and seq_len == len(ed_label_vectors)
                    and seq_len == len(ed_label_ids)
                ):
            raise ValueError(
                f"Expected same lengths, but got:\n"
                f" md_predictions: {len(md_predictions)}\n"
                f" md_labels: {len(md_labels)}\n"
                f" ed_embedding_predictions: {len(ed_embedding_predictions)}\n"
                f" ed_label_vectors: {len(ed_label_vectors)}\n"
                f" ed_label_ids: {len(ed_label_ids)}\n"
            )
        self.doc_len = len(doc_text)
        self.doc_id = doc_id
        self.doc_text = doc_text

        # Find all MD tokens with a 'None' label.
        #  These are non-head WordPiece tokens.
        self.none_pos = []
        for i in range(seq_len):
            if md_labels[i] == IOB_LABEL['None']:
                self.none_pos += [i]

        # Dictionary of span -> EntityLabel
        self.labels = self._parse_labels(
                md_labels, ed_label_vectors, ed_label_ids
            )
        # Dictionary of span -> EntityPrediction
        self.predictions = self._parse_predictions(
                md_predictions, ed_embedding_predictions
            )
        # Compare predictions and labels for MD
        self.md_prediction()

    def _parse_labels(
                self,
                md_labels,
                ed_label_vectors,
                ed_label_ids,
            ):
        """
        Parse labels into dictionary of GtEntity objects with span as key.
        Called once from __init__
        :param md_labels: List of Mention Detection IOB-labels for each token
        :param ed_label_vectors:
            List of ground truth entity vectors for each token
        :param ed_label_ids:
            List of ground truth entity IDs for each token
        """
        # Spans for the input WordPiece tokenized sequences
        tok_spans = []
        span_buffer = []
        for i in range(len(md_labels)):
            self._handle_iob(tok_spans, span_buffer, md_labels[i], i)
        # To empty the buffer, if there's something there
        self._handle_iob(
                tok_spans, span_buffer, IOB_LABEL['O'], len(md_labels)+1
            )

        # Holds list of GtEntity objects
        entities = dict()
        for tok_span in tok_spans:
            # Get span in self.doc_text
            span = self.get_doc_text_span(tok_span)
            mention_text = ' '.join(self.doc_text[span[0]:span[1]+1])
            entities[span] = EntityLabel(
                            span,
                            ed_label_ids[tok_span[0]],
                            ed_label_vectors[tok_span[0]],
                            mention_text,
                        )
        return entities

    def _parse_predictions(self, md_predictions, ed_embeddings):
        """
        Parse MD and ED predictions for this document,
            and generate the self.predictions dictionary of all the
            predicted MD spans and their entity embedding.
        Span is dictionary key.
        Called once from __init__
        :param md_predictions:
            Tensor of logit Mention Detection predictions for each token
        :param ed_embedding_predictions: 
            Tensor of predicted Entity Disambiguation embeddings

        """
        # Class prediction is index of max prediction
        md_predictions = md_predictions.argmax(dim=1)
        # Spans for the input WordPiece tokenized sequences
        tok_spans = []
        span_buffer = []
        for i in range(len(md_predictions)):
            # If the token is not a head WordPiece token
            if i in self.none_pos:
                self._handle_iob(tok_spans, span_buffer, IOB_LABEL['None'], i)
            # Normal (head) WordPiece tokens
            else:
                self._handle_iob(tok_spans, span_buffer, md_predictions[i], i)
        # To empty the buffer, if there's something there
        self._handle_iob(tok_spans, span_buffer, IOB_LABEL['O'], i)
        predictions = dict()
        for tok_span in tok_spans:
            span = self.get_doc_text_span(tok_span)
            mention_text = ' '.join(self.doc_text[span[0]:span[1]+1])
            predictions[span] = EntityPrediction(
                    span,
                    md_predictions[tok_span[0]],  # Use pred of first token
                    ed_embeddings[tok_span[0]],  # Use pred of first token
                    mention_text,
                )
        return predictions

    def get_doc_text_span(self, tokenized_span: Tuple[int]):
        """
        Convert WordPiece span to word-list span.
        Aligns a span for WordPiece tokens to the self.doc_text,
            and returns that span
        :param tokenzied_span: a span of tokens for WordPiece tokens
        :returns: the span for self.doc_text
        """
        span = []
        # Count the number of non-head WordPiece tokens to skip in indexing
        #   the doc_text
        nones = 0
        # Iterate the index of tokenized WordPiece tokens
        for i_tokenized in range(0, tokenized_span[1] + 1):
            # Count indexes to skip so far
            if i_tokenized in self.none_pos:
                nones += 1
            # If we have reached the span, and are on a relevant token
            elif i_tokenized == tokenized_span[0]:
                i_doc_text = i_tokenized - nones
                span += [i_doc_text, i_doc_text]
            # If we are in the span, and on a relevant token
            elif i_tokenized > tokenized_span[0]:
                # Update end of span
                span[1] = i_tokenized - nones
        return tuple(span)

    def md_prediction(self):
        """
        Function sets MD result values in self.predictions
            and self.label_entities by comparing prediction
            spans and label spans.
        Uses "perfect match" for spans, meaning they must have
            the correct start and end.
        Only matches for spans that have a label ID (but not necessarily in KB)
        Does not look at ED prediction.

         * A predicted span is a ...
           * True Positive if predicted span matches the label span
                (a span starting with 'I' is treated as a valid span)
           * False Positive if a predicted span is not a label span
         * A label span is a False Negative if the label span was not predicted
        """
        # Iterate predicted spans
        for pred_span in self.predictions:
            # Search for that span among labels
            if pred_span in self.labels:
                # If that span has a proper label
                if self.labels[pred_span].label_id is not None:
                    # Tell both prediction and GT that they have a match
                    self.predictions[pred_span].correct_md = True
                    self.labels[pred_span].predicted_md = True
            # If there is no such label span (False Positive)
            else:
                self.predictions[pred_span].correct_md = False
        # Iterate the label entities
        for label_entity in self.labels.values():
            # Entities with an ID that have not found a match above
            #  (False Negative)
            if label_entity.predicted_md is None \
                    and label_entity.label_id is not None:
                # Has not been predicted
                label_entity.predicted_md = False

    def get_md_stats(self):
        """
        Returns the number of True Positive, False Positive and False Negative
            MD predictions of this document
        """
        tp = 0
        fp = 0
        fn = 0
        for pred in self.predictions.values():
            # Correct MD predictions are True Positives
            #  Only if this is not a prediction on an unlabeled mention
            if pred.correct_md is True \
                    and self.labels[pred.span].label_id is not None:
                tp += 1
            # Incorrect MD predictions are False Positives
            elif pred.correct_md is False:
                fp += 1
        for label in self.labels.values():
            # Labels that were not predicted are False Negatives
            if label.predicted_md is False:
                fn += 1
        return tp, fp, fn

    def ed_prediction(
                self,
                knowledgebase: KnowledgeBaseWikipedia,
                cand_gen: bool = True,
                use_fallback_for_empty_cs: bool = True,
            ):
        """
        Predict entities with a knowledge base,
            and compare to document labels
        Initializes the ED results in self.predictions and self.labels

         * A predicted span is a ...
           * True Positive if predicted span matches the label span
                (a span starting with 'I' is treated as a valid span),
                AND predicted ED entity matches label entity
           * False Positive if a predicted span is not a label span,
                OR the span was predicted but the predicted ED entity was wrong
         * A label span is a False Negative if the label span was not predicted

        :param knowledgebase: a knowledge base
        :param cand_gen: if True, triggers use of candidate sets generated
            by knowledge base. Entities that are correctly prediced by MD,
            but are not in the KB's candidate gets in_candidate_set = False
        :param use_fallback_for_empty_cs: if True,
            and using candidate generation, using fallback to brute-force
            search over all Wikipedia2vec vectors. This is slower,
            and not recommended during training.
        """
        # First, mark labels that are not in KB
        for label in self.labels.values():
            # Mark labels with a label that is not in the KB
            if label.label_id is not None:
                if not knowledgebase.in_kb(label.label_id):
                    label.in_kb = False
                else:
                    label.label_id = \
                        knowledgebase.get_kb_entity(label.label_id)
                    label.in_kb = True

        # Iterate predictions
        for pred in self.predictions.values():
            # Correct span prediction (MD) is a prerequisite
            if pred.correct_md is True:
                # Get the corresponding EntityLabel
                label = self.labels[pred.span]

                # Skip if label is None (has no ID)
                #  or label is not in KB
                if label.label_id is None or label.in_kb is False:
                    pred.in_kb = False
                    continue

                pred.in_kb = True
                # If tokenizer is provided, get the mention text,
                #  and generate candidates
                if cand_gen:
                    mention_text = pred.mention_text
                else:
                    mention_text = ''

                # Find similar entities
                candidates = knowledgebase.find_similar(
                        pred.ed_embedding_pred,
                        mention_text,
                        use_fallback_for_empty_cs,
                    )

                # If no entities were found by KB,
                #   we treat it as a failed prediction
                if len(candidates) == 0:
                    pred.correct_ed = False
                    label.predicted_ed = False
                    if cand_gen:
                        pred.no_candidates = True
                        label.in_candidate_set = False
                else:
                    candidate_ids = [c[0] for c in candidates]
                    kb_label = knowledgebase.get_kb_entity(label.label_id)
                    # If we used candidate generation,
                    #   and correct label was not in candidate set
                    if cand_gen:
                        pred.candidates = candidate_ids
                        if kb_label not in candidate_ids:
                            pred.correct_ed = False
                            label.predicted_ed = False
                            label.in_candidate_set = False
                            continue
                        else:
                            label.in_candidate_set = True
                    top_pred_id = candidates[0][0]
                    pred.predicted_id = top_pred_id
                    # Finally, if the top prediction is same as label,
                    #   we have a True Positive
                    # TODO: Do this previously in Document-inits?
                    if kb_label == top_pred_id:
                        pred.correct_ed = True
                        label.predicted_ed = True
                    # Otherwise, a False Positive and a False Negative
                    else:
                        pred.correct_ed = False
                        label.predicted_ed = False
            # Incorrectly predicted span gives False Positive
            elif pred.correct_md is False:
                pred.correct_ed = False

        # Finally, find the False Negatives
        for label in self.labels.values():
            # labels with an ID that were not matched above are False Negatives
            if label.label_id is not None \
                    and label.in_kb is True \
                    and label.predicted_ed is None:
                label.predicted_ed = False

    def get_ed_stats(self):
        """
        After running ed_prediction, this gives the number of
            True Positives, False Positives, False Negatives,
            labels not in recent knowledge base, and labels not in candidates

        :returns: five ints for number of
            True Positives, False Positives, and False Negatives,
            labels not in recent knowledge base,
            labels not in candidate sets (0 if candidate sets not used)
        """
        tp = 0
        fp = 0
        fn = 0
        not_in_kb = 0
        not_in_cand = 0
        no_cands = 0
        # Iterate predictions for True Positives and False Positives
        for pred in self.predictions.values():
            # Case: Correct_MD is True, but the corresponding label is None
            #  Do nothing
            if pred.correct_ed is None:
                continue
            # True Positive if the ED prediction was correct
            elif pred.correct_ed is True:
                tp += 1
            # False Positive if the ED prediction was wrong
            elif pred.correct_ed is False:
                fp += 1
                if pred.no_candidates is True:
                    no_cands += 1
        # Iterate labels for False Negatives, labels not in KB,
        #  and labels not in candidate sets
        for label in self.labels.values():
            # False Negative for labels not predicted
            if label.predicted_ed is False:
                fn += 1
                # Mark if labels was not in candidate set
                if label.in_candidate_set is False:
                    not_in_cand += 1
            # Labels not in KB are used as False Negative
            #   only for "Out-Of-KB evaluation"
            if label.in_kb is False:
                not_in_kb += 1
        return tp, fp, fn, not_in_kb, not_in_cand, no_cands

    def __str__(self):
        """
        Return a string representation of the document
            with annotated labels and predictions
        """
        # Idea:
        # All predictions are marked with {{}},
        #   and all labels that are not predictions are marked with [[]]
        # Correct predictions (TP) have green {{}}, wrong (FP) are red
        # Unpredicted labels (FN) have red [[]]
        # For TP and FN, the predicted / label ID respectively like
        #  ((Q33)), with red for wrong and green for correct

        words = self.doc_text
        # ANSI colors
        c = {
                'e': "\033[0;0m",  # End
                'y': "\033[33m",   # Green
                'g': "\033[32m",   # Green
                'r': "\033[31m",   # Red
            }
        # MD True Positive Tag
        md_tp_s = c['g'] + '{{' + c['e']
        md_tp_e = c['g'] + '}}' + c['e']
        # MD False Positive Tag
        md_fp_s = c['r'] + '{{' + c['e']
        md_fp_e = c['r'] + '}}' + c['e']
        # MD False Negative Tag
        md_fn_s = c['r'] + '[[' + c['e']
        md_fn_e = c['r'] + ']]' + c['e']
        # ED True Positive Tag for label info
        ed_tp_s = c['g'] + '(('
        ed_tp_e = '))' + c['e']
        # ED False Start Tag
        ed_fp_s = c['r'] + '(('
        # ED False End Tag
        ed_fp_e = '))' + c['e']
        # Entity not in KB/CS/Empty CS:
        err_s = c['y'] + '[['
        err_e = ']]' + c['e']

        # Parse predictions
        for span, pred in self.predictions.items():
            # MD TP:
            if pred.correct_md is True:
                # MD TP Start Tag on first word
                words[span[0]] = md_tp_s + words[span[0]]
                # ED TP: Entity ID in Green (())
                if pred.correct_ed is True:
                    words[span[1]] += ed_tp_s + pred.predicted_id + ed_tp_e
                # ED False (Negative and Positive)
                elif pred.correct_ed is False:
                    label = self.labels[span]
                    # GT entity not in KB
                    pred_id = pred.predicted_id
                    if pred_id is None:
                        pred_id = 'None'
                    words[span[1]] += ed_fp_s + pred_id \
                        + ' != ' + label.label_id + ed_fp_e
                    if pred.no_candidates is True:
                        words[span[1]] += err_s + 'CS is Ø' + err_e
                    elif label.in_candidate_set is False:
                        words[span[1]] += err_s + 'Not in CS' + err_e
                # MD TP End Tag on last word
                words[span[1]] += md_tp_e
            # MD FP: No ED-stuff
            elif pred.correct_md is False:
                words[span[0]] = md_fp_s + words[span[0]]
                words[span[1]] += md_fp_e

        # MD FN
        for span, label in self.labels.items():
            if label.in_kb is False:
                words[span[1]] += err_s + 'Not in KB' + err_e
            # MD FN
            if label.predicted_md is False:
                words[span[0]] = md_fn_s + words[span[0]]
                words[span[1]] += ed_fp_s + label.label_id + ed_fp_e + md_fn_e
        return ' '.join(words)

    @staticmethod
    def _handle_iob(spans, buffer, value, i):
        """
        Logic to handle an IOB label or prediction by
           adding to span or buffer
        :param spans: list of parsed spans
        :param buffer: the current span, as being worked on
        :param value: the IOB value of the current token
        :param i: the current position in the document
        """
        # If 'B'
        if value == IOB_LABEL['B']:
            # Add current buffer to spans
            if len(buffer) > 0:
                spans += [(buffer[0], buffer[-1])]
            # Clear buffer
            buffer.clear()
            # Add current to buffer
            buffer += [i]
        # If 'I', add to buffer
        elif value == IOB_LABEL['I'] and not len(buffer) == 0:
            # if len(buffer) == 0:
            #     print(f"Lone 'I'!")
            buffer += [i]
        # If None, this is a non-head WordPiece token
        elif value == IOB_LABEL['None']:
            # If buffer is not empty, and this follows a token in the buffer
            if len(buffer) > 0 and i - 1 == buffer[-1]:
                # Add to buffer
                buffer += [i]
        # If 'O'
        elif value == IOB_LABEL['O']:
            # If there is a span in buffer
            if len(buffer) > 0:
                # Add current buffer to span
                spans += [(buffer[0], buffer[-1])]
                # Clear buffer
                buffer.clear()


def get_document_from_sequences(
            doc_id: int,
            positions: List[Tuple],
            md_preds: Tensor,
            md_labels: Tensor,
            ed_preds: Tensor,
            ed_label_vectors: Tensor,
            ed_label_ids: List,
            doc_text: List[str],
        ):
    """
    Given a set of sequences belonging to the same document, this function
     resolves overlaps, and handles tie-breaking of predictions.
     The tie-breaker is the MD prediction with the strongest magnitude.

    :param doc_id: ID of the current processed document
    :param positions: list of tuples with the position of each input sequence
        from the origin document
    :param md_preds:
        Tensor of sequenes of logit Mention Detection predictions
            for each token in each sequence
    :param md_labels:
        List of sequences of Mention Detection IOB-labels
            for each token in each sequence
    :param ed_preds:
        Tensor of sequences of predicted Entity Disambiguation embeddings
    :param ed_label_vectors:
        List of sequences of ground truth entity vectors
            for each token in each sequence
    :param ed_label_ids:
        List of sequences of ground truth entity IDs
            for each token in each sequence
    :param doc_text: List of the original word tokens for this document
        Necessary to get the right mention text. Passed to document as is.
    :returns: a Document object, where overlapping sequences have been resolved
        by the magnitude of MD predictions
    """
    # Handle first sequence
    pos_curr = positions[0]
    # Start with full first sequence, excluding [CLS], [SEP] and [PAD]
    md_preds_res = md_preds[0][1:pos_curr[1]+1]
    md_labels_res = md_labels[0][1:pos_curr[1]+1]
    ed_preds_res = ed_preds[0][1:pos_curr[1]+1]
    ed_label_vectors_res = ed_label_vectors[0][1:pos_curr[1]+1]
    ed_label_ids_res = ed_label_ids[0][1:pos_curr[1]+1]

    n_data_points = len(positions)
    # Deal with subsequent sequences from current document
    for i_pos in range(1, n_data_points):
        # Document position of previous sequence
        pos_prev = positions[i_pos - 1]
        # Document position of current sequence
        pos_curr = positions[i_pos]
        # Find overlap (if previous sequence ends after start of current)
        n_overlap = pos_prev[1] - pos_curr[0]
        if n_overlap < 0:
            raise ValueError(
                    f"Subsequent sequences are not overlapping or "
                    f"adjacent. Got an overlap of {n_overlap} from "
                    f"document {doc_id}, sequences {i_pos-1} and "
                    f"{i_pos}, with positions {pos_prev}, {pos_curr}"
                )

        seq_len = pos_curr[1] - pos_curr[0]

        # Current sequence predictions, excluding [CLS], [SEP], [PAD]
        md_preds_curr = md_preds[i_pos][1:1+seq_len]
        # Overlap from current sequence ("contenders")
        contenders = md_preds_curr[:n_overlap]
        # Overlap from previous sequence ("defenders")
        defenders = md_preds_res[-n_overlap:]
        for i_o, tup in enumerate(zip(contenders, defenders)):
            contender = tup[0]
            defender = tup[1]
            # Tie breaker: Pick the prediction with the highest
            #   magnitude MD prediction for any class
            if contender.amax() > defender.amax():
                md_preds_res[- n_overlap + i_o] = contender
                # Same for ED (MD prediction resolves ED)
                ed_preds_res[- n_overlap + i_o] = \
                    ed_preds[i_pos][1 + i_o]

        # Part of sequence that's not [CLS], [SEP], [PAD]
        md_labels_curr = md_labels[i_pos][1: 1 + seq_len]
        # Add what's not overlapping from current sequence
        md_labels_res = cat(
                (md_labels_res, md_labels_curr[n_overlap:])
            )
        md_preds_res = cat(
                (md_preds_res, md_preds_curr[n_overlap:])
            )

        # Same for ED
        # Part of sequence that's not [CLS], [SEP], [PAD]
        ed_preds_curr = ed_preds[i_pos][1: 1 + seq_len]
        ed_labels_curr = ed_label_ids[i_pos][1: 1 + seq_len]
        # Add what's not overlapping from current sequence
        ed_preds_res = cat(
                (ed_preds_res, ed_preds_curr[n_overlap:])
            )
        ed_label_vectors_curr = ed_label_vectors[i_pos][1: 1 + seq_len]
        ed_label_vectors_res = cat(
                (ed_label_vectors_res, ed_label_vectors_curr[n_overlap:])
            )
        ed_label_ids_res += ed_labels_curr[n_overlap:]
    document = Document(
            doc_id,
            doc_text,
            md_preds_res,
            md_labels_res,
            ed_preds_res,
            ed_label_vectors_res,
            ed_label_ids_res,
        )
    return document

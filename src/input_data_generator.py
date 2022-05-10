"""
This class generates tokenized input data for BERT from CoNLL documents
and exctracts their labels
"""

import time
from typing import Tuple, List
from os.path import isfile
from math import ceil
from torch import cat, Tensor, LongTensor

from src.conll_document import ConllDocument
from src.toolbox import get_tokenizer, get_docs
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia


class InputDataGenerator:
    # Mapping from character label to numerical label
    IOB_LABEL = {
                'I': 0,
                'O': 1,
                'B': 2,
                'None': 3
            }

    def __init__(
                self,
                knowledgebase: KnowledgeBaseWikipedia,
                tokenizer_pretrained_id: str = 'bert-base-uncased'
            ):
        """
        :param knowledgebase: a KnowledgeBase object
        :param tokenizer_pretrained_id: the URI of the Huggingface tokenizer
        """
        self.kb = knowledgebase

        self.tokenizer = get_tokenizer(tokenizer_pretrained_id)

        # Token IDS for three special tokens
        self.CLS_ID = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.SEP_ID = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.PAD_ID = self.tokenizer.convert_tokens_to_ids('[PAD]')
        # IOB labels as integers
        entity_embedding_size = self.kb.dim_size
        self.EMPTY_EMBEDDING = Tensor([0] * entity_embedding_size)

    def generate_for_conll_doc(
                self,
                document: ConllDocument
            ) -> Tuple[List[Tensor], Tuple[int]]:
        """
        Function yields the three input vectors to BERT
        for a given CoNLL document.
        Documents tokenized to more than 510 tokens
        are split into multiple sequences:
            510 tokens at a time are placed into datapoints, where the last
            data point also has the last 510 tokens (overlap in last two).

        :param document: a ConllDocument object
        :returns:
        A list of data points, where each data point is made up of:
            * three input LongTensors with input_ids, attention_mask,
                and token_type_ids; and
            * two Tensors of:
                * a one-hot IOB mention detection labels,
                * the Wikipedia2vec embeddings of 'B' tokens,
                    being the Entity Disambiguation label.
        And a list of the position of each sequence in the tokenized
            input document
        """
        # Return variable with output tensors
        data_points = []
        # The tokenized document
        doc_tokens = []
        # The IOB Mention Detection labels for each document token
        doc_md_labels = []
        # The Wikidata QIDs of 'B'-tokens
        doc_ed_labels = []
        # The Wikipedia2vec embedding of labeled 'B'-tokens
        doc_ed_embeddings = []

        # Tokenize the document, and collect labels
        for token in document.tokens:
            word = token.text
            label = token.true_label
            if word.isupper():
                word = word.capitalize()
            word_tokenized = self.tokenizer.encode(
                    word,
                    add_special_tokens=False
                )
            doc_tokens += word_tokenized

            # Deal with IOB mention detection labels
            # Don't consider 'B' mentions without an entity label
            if label == 'O':
                md_label_vector = self.IOB_LABEL['O']
            elif label == 'I':
                md_label_vector = self.IOB_LABEL['I']

                # If previous label is 'O',
                #  this is from a mention without an entity label
                # if doc_md_labels[-1] == self.IOB_LABEL['O']:
                #     md_label_vector = self.IOB_LABEL['O']
                # # Else, it's a entity-labeled mention
                # else:
                #     md_label_vector = self.IOB_LABEL['I']
            else:
                md_label_vector = self.IOB_LABEL['B']

            # Only head WordPiece token has IOB label
            doc_md_labels += [md_label_vector]
            none_vector = self.IOB_LABEL['None']
            doc_md_labels += [none_vector] * (len(word_tokenized) - 1)

            # Deal with ED labels (Wikipedia2vec embedding vectors)
            if label not in ['I', 'O', 'B']:
                doc_ed_labels += [label] + [None] * (len(word_tokenized) - 1)
                # print(label)
                entity_vec = self.kb.get_entity_vector(label)
                if entity_vec is None:
                    entity_vec = self.EMPTY_EMBEDDING
                doc_ed_embeddings += [Tensor(entity_vec)] \
                    + [self.EMPTY_EMBEDDING] * (len(word_tokenized) - 1)
            else:
                doc_ed_labels += [None] * len(word_tokenized)
                doc_ed_embeddings += \
                    [self.EMPTY_EMBEDDING] * len(word_tokenized)

        max_sequence_len = self.tokenizer.max_len_single_sentence
        sequence_len = len(doc_tokens)
        # Minimum number of split data points for this document
        n_data_points = ceil(sequence_len / max_sequence_len)
        # Start and end of each data point
        start_pos = 0
        end_pos = min(max_sequence_len, sequence_len)
        # Calculate number of tokens of overlap
        overlap = 0
        if n_data_points > 1:
            overlap = int(
                    (max_sequence_len - sequence_len % max_sequence_len)
                    /
                    (n_data_points-1)
                )
        # Keep track of the position of each sequence in the full document
        sequence_pos = []
        # Generate the split documents
        for i_data_point in range(n_data_points):
            # Build the various data vectors
            input_ids = [self.CLS_ID] \
                 + doc_tokens[start_pos:end_pos] \
                 + [self.SEP_ID]

            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * 512
            md_labels = [self.IOB_LABEL['None']] \
                + doc_md_labels[start_pos:end_pos] \
                + [self.IOB_LABEL['None']]

            ed_labels = [None] + doc_ed_labels[start_pos:end_pos] + [None]
            ed_embeddings = [self.EMPTY_EMBEDDING] \
                + doc_ed_embeddings[start_pos:end_pos] \
                + [self.EMPTY_EMBEDDING]

            sequence_pos += [(start_pos, end_pos)]

            # Move pointers to next sub-sequence using ideal overlap
            # start_pos += max_sequence_len - overlap
            end_pos = min(sequence_len, end_pos + max_sequence_len - overlap)
            # Move start for maximum overlap
            start_pos = end_pos - max_sequence_len

            # Padding for short documents
            if len(input_ids) < (max_sequence_len + 2):
                pad_len = max_sequence_len + 2 - len(input_ids)
                input_ids += [self.PAD_ID] * pad_len
                attention_mask += [0] * pad_len
                md_labels += [self.IOB_LABEL['None']] * pad_len
                ed_labels += [None] * pad_len
                ed_embeddings += [self.EMPTY_EMBEDDING] * pad_len

            input_ids = LongTensor(input_ids).unsqueeze(0)
            attention_mask = LongTensor(attention_mask).unsqueeze(0)
            token_type_ids = LongTensor(token_type_ids).unsqueeze(0)
            md_labels = LongTensor(md_labels).unsqueeze(0)
            ed_embeddings = \
                cat([t.unsqueeze(0) for t in ed_embeddings]).unsqueeze(0)

            data_points.append(
                    (
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        md_labels,
                        ed_embeddings,
                        ed_labels
                    )
                )

        return data_points, sequence_pos

    def generate_for_docs(
                self,
                docs,
                progress: bool = False,
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[Tuple]]:
        """
        :param docs: an iterator over a file with each line a document
        :param progress: print progress
        :returns: five torch.Tensor and a list of tuples containing
            the document number of each data point,
            and its sequence position in the document
        """
        data = []
        doc_indices = []
        t0 = time.time()

        if progress:
            print()

        for i_doc, doc in enumerate(docs):
            if progress and i_doc != 0 and i_doc % 100 == 0:
                print(f"Processed {i_doc} documents ", end='')
                avg_doc_time = (time.time() - t0) / i_doc
                if avg_doc_time >= 1:
                    print(f"({avg_doc_time:.1f} s/doc)", end='\r')
                else:
                    print(f"({1/avg_doc_time:.1f} doc/s)", end='\r')

            data_points, doc_positions = self.generate_for_conll_doc(doc)
            data += data_points
            doc_indices += [(i_doc, pos) for pos in doc_positions]

        if progress:
            print()

        # Make tensors of the whole dataset
        input_ids = cat([d[0] for d in data])
        attention_mask = cat([d[1] for d in data])
        token_type_ids = cat([d[2] for d in data])
        md_labels = cat([d[3] for d in data])
        ed_embeddings = cat([d[4] for d in data])
        ed_labels = [d[5] for d in data]

        return input_ids, attention_mask, token_type_ids, \
            md_labels, ed_embeddings, ed_labels, doc_indices

    def generate_for_file(self, file: str, progress: bool = False):
        """
        Generates tokenized BERT input vectors for all documents in file

        :param file: file path to an annotated file
        :param progress: print progress
        :returns: five torch.Tensor and a list of tuples containing
            the document number of each data point,
            and its sequence position in the document
        """

        if not isfile(file):
            raise FileNotFoundError(f"No file at '{file}'")

        docs = get_docs(file)
        return self.generate_for_docs(docs, progress)

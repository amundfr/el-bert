"""
Script to make a light-weight version of Wikipedia2vec,
with entities above a certain score.

Run as a module to avoid import problems:

    python -m scripts.compact_kb -h

"""

import argparse
import six
import numpy as np
from wikipedia2vec import Wikipedia2Vec
from wikipedia2vec.dictionary import Dictionary
from marisa_trie import Trie, RecordTrie
from collections import defaultdict
from os import path, mkdir
from src.toolbox import get_docs
# from urllib.parse import unquote
# from unidecode import unidecode
from configparser import ConfigParser


parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "--wikipedia2vec", type=str,
        default='/ex_data/knowledgebases/wikipedia2vec/' +
                'enwiki_20180420_100d.pkl',
        help="path to Wikipedia2vec .pkl file"
    )
parser.add_argument(
        "--min_score", type=int, default=0,
        help="threshold of Wikipedia2vec score for entities in KB"
    )
parser.add_argument(
        "--add_conll_entities", action='store_true', default=False,
        help="add conll entities, regardless of score. " +
             "Reads CoNLL file path from config.ini"
    )
parser.add_argument(
        "output_dir", type=str,
        help="path to directory to store KB files"
    )
args = parser.parse_args()

config = ConfigParser()
config.read('config.ini')

wikipedia2vec = Wikipedia2Vec.load(args.wikipedia2vec)
save_folder = args.output_dir
suffix = str(args.min_score)
suffix += '_with_conll' if args.add_conll_entities else ''

if not path.isdir(save_folder):
    mkdir(save_folder)

# Construct the file path for the entities that are and are not in the KB
wikipedia2vec_name = path.split(args.wikipedia2vec)[1]
wikipedia2vec_name = wikipedia2vec_name.split('.')[0]

not_in_w2v = 0
if args.add_conll_entities:
    conll_file = config['DATA']['Annotated Dataset']
    conll_labels = set([
            tok.true_label
            for doc in get_docs(conll_file)
            for tok in doc.tokens if tok.true_label not in ['I', 'O', 'B']
        ])
    conll_ents = []
    # Get the expected Wikipedia2vec name of the entities
    for conll_ent in conll_labels:
        w2v_ent = wikipedia2vec.get_entity(conll_ent.replace('_', ' '))
        if w2v_ent:
            conll_ents += [w2v_ent.title]
        else:
            not_in_w2v += 1
print(f"Not in Wikipedia2vec: {not_in_w2v}. Expected 254")

# Wikipedia2vec stuff:
# Dictionary of words (empty)
words = defaultdict(int)
# Dictionary of entity to index in vectors list
entities = defaultdict(int)
# List of entity stats
entity_stats = []

# List of entity vectors
vectors = []
# Counter of entities
n = 0

# Keeping some statistics of entities
n_under_score = 0
duplicates = 0
no_vector = 0
conll_exception = 0
# Find the entities from the mapping that are in wikipedia2vec
for i_entity, entity in enumerate(wikipedia2vec.dictionary.entities()):
    if (i_entity % 100000) == 0:
        print(f"  {i_entity} entities processed, {n} entities added", end='\r')
    entity_title = entity.title  # unidecode(unquote(entity.title))
    if entity_title in entities:
        duplicates += 1
        if entity.count <= entity_stats[entities[entity_title]][0]:
            continue

    # If score is too low (and entity is not in conll)
    if entity.count < args.min_score:
        if args.add_conll_entities:
            if entity_title not in conll_ents:
                n_under_score += 1
                continue
            else:
                conll_exception += 1
        else:
            n_under_score += 1
            continue

    vector = wikipedia2vec.get_entity_vector(
            entity.title,
            resolve_redirect=True,
        )
    if len(vector) != 100:
        print(f"\nVector None: {entity.title}")
    vectors.append(np.array(vector, dtype=np.float32))
    entities[entity_title] = n
    entity_stats += [np.array([entity.count, entity.doc_count])]
    n += 1
    # Write to mapping file
print(
        f"\nAdded: {len(entities)}, Too low score: {n_under_score}, "
        f"No vector: {no_vector}, Skipped duplicates: {duplicates}\n"
        f"Added as a CoNLL exception: {conll_exception}\n"
      )
syn0 = np.empty((len(vectors), vectors[0].size))
word_dict = Trie(words.keys())
entity_dict = Trie(entities.keys())
redirect_dict = RecordTrie('<I')

for (word, ind) in six.iteritems(word_dict):
    syn0[ind] = vectors[words[word]]

entity_offset = len(word_dict)
for (title, ind) in six.iteritems(entity_dict):
    syn0[ind + entity_offset] = vectors[entities[title]]

word_stats = np.zeros((len(word_dict), 2), dtype=np.int32)
entity_stats = np.array(entity_stats, dtype=np.int32)

dictionary = Dictionary(
        word_dict, entity_dict, redirect_dict, word_stats, entity_stats,
        None, False, dict()
    )
ret = Wikipedia2Vec(dictionary)
ret.syn0 = syn0
ret.syn1 = None

wiki2vec_file = wikipedia2vec_name + '.pkl'
wiki2vec_file = path.join(save_folder, wiki2vec_file)
ret.save(wiki2vec_file)
print(f'\nDone!\n Wrote Wikipedia2vec to {wiki2vec_file}\n')

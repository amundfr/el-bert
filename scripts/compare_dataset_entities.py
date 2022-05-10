"""
To evaluate the advantage of a secondary training set on CoNLL evaluation,
this script compares the overlap of entities between CoNLL and a
Wikipedia dataset.

python -m scripts.compare_dataset_entities -h
"""

import re
import argparse
import configparser
import os.path
from collections import defaultdict
from src.toolbox import get_docs
from src.knowledge_base_wikidata import KnowledgeBaseWikidata as KbWikidata


parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
w_file = '/ex_data/annotated_datasets/train_articles_dataset_score_0_50k.txt'
parser.add_argument(
        "--wikipedia_dataset_file", type=str,
        default=w_file,
        help="path to text file with annotated dataset"
    )
parser.add_argument(
        "--conll_dataset_file", type=str,
        default='/ex_data/annotated_datasets/conll-wikidata-iob-annotations',
        help="path to text file with CoNLL anotated dataset"
    )
parser.add_argument(
        "--mentions", action='store_true',
        default=False,
        help="flag: if provided, count mentions rather than unique entities",
    )

args = parser.parse_args()
config = configparser.ConfigParser()
config.read('config.ini')

kb_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']
# Load full Wikidata KB, but not with CG
kb_wd = KbWikidata(
    os.path.join(kb_dir, config['KNOWLEDGE BASE']['Wikipedia2vec File']),
    os.path.join(kb_dir, config['KNOWLEDGE BASE']['Wikidata To Wikipedia'])
)

wiki_split = [48500, 500, 1000]
wiki_ents = defaultdict(list)
not_in_kb_wd = []
for i_wiki, doc in enumerate(open(args.wikipedia_dataset_file)):
    if i_wiki < wiki_split[0]:
        key = 'train'
    elif i_wiki < (wiki_split[0] + wiki_split[1]):
        key = 'val'
    else:
        key = 'test'
    for s in re.findall(r'(\\\?\\Q\d+)', doc):
        wd_ent = s.replace('\\?\\', '')
        if kb_wd.in_kb(wd_ent):
            wiki_ents[key] += [wd_ent]
        else:
            not_in_kb_wd += [wd_ent]
print(f"\n{len(not_in_kb_wd)} Wikipedia Articles entities not "
      f"in current Wikidata KB (expected 0)\n")

conll_split = [946, 216, 231]
conll_ents = defaultdict(list)
for i_conll, doc in enumerate(get_docs(args.conll_dataset_file)):
    if i_conll < conll_split[0]:
        key = 'train'
    elif i_conll < (conll_split[0] + conll_split[1]):
        key = 'val'
    else:
        key = 'test'
    for token in doc.tokens:
        ent = token.true_label
        if ent not in ['I', 'O', 'B']:
            conll_ents[key] += [ent]

if args.mentions:
    wiki_sets = wiki_ents
    conll_sets = conll_ents
else:
    wiki_sets = dict((k, set(v)) for k, v in wiki_ents.items())
    conll_sets = dict((k, set(v)) for k, v in conll_ents.items())

# Unique entities in each dataset partition
wt = len(wiki_sets['test'])
wv = len(wiki_sets['val'])
wtr = len(wiki_sets['train'])
ct = len(conll_sets['test'])
cv = len(conll_sets['val'])
ctr = len(conll_sets['train'])
c = len(set(
        list(conll_sets['test'])
        + list(conll_sets['val'])
        + list(conll_sets['train'])
        ))
w = len(set(
        list(wiki_sets['test'])
        + list(wiki_sets['val'])
        + list(wiki_sets['train'])
        ))

# Intersect of entities in two dataset partitions
ct_wt = [c for c in conll_sets['test'] if c in wiki_sets['test']]
ct_wv = [c for c in conll_sets['test'] if c in wiki_sets['val']]
ct_wtr = [c for c in conll_sets['test'] if c in wiki_sets['train']]
cv_wt = [c for c in conll_sets['val'] if c in wiki_sets['test']]
cv_wv = [c for c in conll_sets['val'] if c in wiki_sets['val']]
cv_wtr = [c for c in conll_sets['val'] if c in wiki_sets['train']]
ctr_wt = [c for c in conll_sets['train'] if c in wiki_sets['test']]
ctr_wv = [c for c in conll_sets['train'] if c in wiki_sets['val']]
ctr_wtr = [c for c in conll_sets['train'] if c in wiki_sets['train']]

stats_of = "mentions:" if args.mentions else "unique entities:"
print("Stats of " + stats_of)
# Simple version, with only Wikiarticles Train
# print(f" {'Conll (v)':9} | {'':6} | {'Wiki':<6}")
# print(f" {'':9} | {'':6} | {'Train':<6}")
# print(f" {'':9} | {'':6} | {wtr:>6}")
# print(f" {'Train ':<9} | {ctr:>6} | {len(ctr_wtr):>6}")
# print(f" {'Val   ':<9} | {cv:>6} | {len(cv_wtr):>6}")
# print(f" {'Test  ':<9} | {ct:>6} | {len(ct_wtr):>6}")

# Complete version
print(f" {'Conll (v)':9} | {'':6} | {'Wiki':<6} | {'Wiki':<6} | {'Wiki':<6}")
print(f" {'':9} | {'':6} | {'Test':<6} | {'Val':<6} | {'Train':<6} | All")
print(f" {'':9} | {'':6} | {wt:>6} | {wv:>6} | {wtr:>6} | {w:>6}")
print(f""" {'Train ':<9} | {ctr:>6} | {len(ctr_wt):>6} | \
{len(ctr_wv):>6} | {len(ctr_wtr):>6}""")
print(f""" {'Val   ':<9} | {cv:>6} | {len(cv_wt):>6} | \
{len(cv_wv):>6} | {len(cv_wtr):>6}""")
print(f""" {'Test  ':<9} | {ct:>6} | {len(ct_wt):>6} | \
{len(ct_wv):>6} | {len(ct_wtr):>6}""")
print(f""" {'Total ':<9} | {c:>6}""")

# Entities in CoNLL dataset partitions, but not in CoNLL train
nctr_ct = sum(1 for c in conll_sets['test'] if c not in conll_sets['train'])
nctr_cv = sum(1 for c in conll_sets['val'] if c not in conll_sets['train'])

# Entities in Wiki dataset partitions, but not in CoNLL train
nctr_wt = sum(1 for c in wiki_sets['test'] if c not in conll_sets['train'])
nctr_wv = sum(1 for c in wiki_sets['val'] if c not in conll_sets['train'])
nctr_wtr = sum(1 for c in wiki_sets['train'] if c not in conll_sets['train'])

# Intersect of two sets of entities not in CoNLL train
nctr_ct_wt = sum(1 for c in ct_wt if c not in conll_sets['train'])
nctr_ct_wv = sum(1 for c in ct_wv if c not in conll_sets['train'])
nctr_ct_wtr = sum(1 for c in ct_wtr if c not in conll_sets['train'])
nctr_cv_wt = sum(1 for c in cv_wt if c not in conll_sets['train'])
nctr_cv_wv = sum(1 for c in cv_wv if c not in conll_sets['train'])
nctr_cv_wtr = sum(1 for c in cv_wtr if c not in conll_sets['train'])

print()
print("Mentions" if args.mentions else "Entities" + " not in CoNLL Train:")
print("""(Pretraining on 'Wiki train' gives an advantage on these {}
 in comparison to only training on CoNLL train)""".format(
     "Mentions" if args.mentions else "Entities"
    ))

# Simple version, with only Wikiarticles Train
print(f" {'Conll (v)':9} | {'':6} | {'Wiki':<6}")
print(f" {'':9} | {'':6} | {'Train':<6}")
print(f" {'':9} | {'':6} | {nctr_wtr:>6}")
print(f" {'Val   ':<9} | {nctr_cv:>6} | {nctr_cv_wtr:>6}")
print(f" {'Test  ':<9} | {nctr_ct:>6} | {nctr_ct_wtr:>6}")

# Complete version
# print(f" {'Conll (v)':9} | {'':6} | {'Wiki':<6}")
# print(f" {'':9} | {'':6} | {'Test':<6} | {'Val':<6} | {'Train':<6}")
# print(f" {'':9} | {'':6} | {nctr_wt:>6} | {nctr_wv:>6} | {nctr_wtr:>6}")
# print(f""" {'Test  ':<9} | {nctr_ct:>6} | {nctr_ct_wt:>6} | \
# {nctr_ct_wv:>6} | {nctr_ct_wtr:>6}""")
# print(f""" {'Val   ':<9} | {nctr_cv:>6} | {nctr_cv_wt:>6} | \
# {nctr_cv_wv:>6} | {nctr_cv_wtr:>6}""")

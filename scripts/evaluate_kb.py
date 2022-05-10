"""
Evaluate KB: Iterate the CoNLL datasets and find Wikipedia2vec KB's coverage
Call like a module to avoid "ModuleNotFoundError: No module named 'src'":
    python3 -m scripts.evaluate_kb -h
"""

import configparser
import argparse
import os.path
from src.toolbox import get_docs, get_knowledge_base
from src.knowledge_base_wikidata import KnowledgeBaseWikidata
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia


parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "--conll", type=str,
        default='/ex_data/annotated_datasets/conll-wikipedia-iob-annotations',
        help="path to a CoNLL file"
    )
parser.add_argument(
        "--kb_type", type=str, nargs=1, default='pedia',
        choices=['pedia', 'data'],
        help="which Knowledge Base TYPE to use. "
    )
parser.add_argument(
        "--compact_kb", action='store_true', default=False,
        help="Knowledge Base for Candidate Generation or not. "
    )

args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

wiki = args.kb_type[0]
print(f"""Evaluating KB with base Wiki{wiki} 
    to use with{"" if args.compact_kb else "out"} Candidate Generation""")

if args.compact_kb:
    if wiki == 'pedia':
        kb = KnowledgeBaseWikipedia(
            os.path.join(
                config["KNOWLEDGE BASE"]["Compact Wikipedia KB"],
                config["KNOWLEDGE BASE"]["Wikipedia2vec File"]
            )
        )
    elif wiki == 'data':
        kb = KnowledgeBaseWikidata(
            os.path.join(
                config["KNOWLEDGE BASE"]["Compact Wikidata KB"],
                config["KNOWLEDGE BASE"]["Wikipedia2vec File"]
            ),
            os.path.join(
                config["KNOWLEDGE BASE"]["Compact Wikidata KB"],
                config['KNOWLEDGE BASE']['Wikidata To Wikipedia']
            )
        )
    else:
        raise ValueError(f"Expected kb_type = 'pedia' or 'data'; got {wiki}")
else:
    kb = get_knowledge_base(config, wiki, with_cg=args.compact_kb)

# get conll docs
docs = list(get_docs(args.conll))
n = len(docs)
not_in_kb = []
in_kb = []
# iterate docs
for i_doc, doc in enumerate(docs):
    # Iterate document tokens
    for token in doc.tokens:
        word = token.text
        label = token.true_label
        if label not in ['I', 'O', 'B']:
            if kb.in_kb(label):
                in_kb += [label]
            else:
                not_in_kb += [label]
print()

n_kb = len(in_kb)
n_not_kb = len(not_in_kb)
mentions = n_kb + n_not_kb
unique_in_kb = len(set(in_kb))
unique_not_in_kb = len(set(not_in_kb))
unique = unique_in_kb + unique_not_in_kb

print(
        f"\n Results:\n"
        f"   {mentions:>6} entity mentions in total\n"
        f"   {n_kb:>6} of mentions are in the KB\n"
        f"   {n_not_kb:>6} of mentions are NOT in the KB\n"
        f"   {100*n_kb/mentions:>6.2f} % coverage\n\n"
    )
print(
        f"   {unique:>6} unique entities mentioned in total\n"
        f"   {unique_in_kb:>6} of unique entities are in KB\n"
        f"   {unique_not_in_kb:>6} of unique entities are not in KB\n"
        f"   {100*unique_in_kb/unique:>6.2f} % coverage\n\n"
    )
print(
        f"   {kb.n_entities():>6} entities in the knowledgebase"
        f"\n\n"
    )

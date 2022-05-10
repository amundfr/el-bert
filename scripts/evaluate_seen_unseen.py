"""
Script to compare the evaluation results on an AIDA-CoNLL dataset
of a model pre-trained on the Wikipedia Articles dataset

Run as module:
    python -m scripts.evaluate_seen_unseen -h
"""

from collections import defaultdict
import configparser
import argparse
import re
from os import path

from src.knowledge_base_wikidata import KnowledgeBaseWikidata
from src.toolbox import get_docs


config = configparser.ConfigParser()
config.read('config.ini')

parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "pred_file", type=str,
        help="path to a (In-KB) predictions .tsv file"
    )
parser.add_argument(
        "gt_file", type=str,
        help="path to a (In-KB) ground truth .tsv file"
    )
w_file = '/ex_data/annotated_datasets/train_articles_dataset_score_0_50k.txt'
parser.add_argument(
        "--wiki_file", type=str,
        default=w_file,
        help="path to text file with annotated Wikipedia Articles dataset"
    )
parser.add_argument(
        "--conll_file", type=str,
        default=config['DATA']['Annotated Dataset'],
        help="path to text file with CoNLL anotated dataset"
    )

args = parser.parse_args()

kb_wd = KnowledgeBaseWikidata(
        path.join(config['KNOWLEDGE BASE']['Wikipedia2vec Directory'],
                  config['KNOWLEDGE BASE']['Wikipedia2vec File']),
        path.join(config['KNOWLEDGE BASE']['Wikipedia2vec Directory'],
                  config['KNOWLEDGE BASE']['Wikidata To Wikipedia']),
    )

wiki_split = [48500, 500, 1000]
wiki_ents = defaultdict(list)
not_in_kb_wd = []
for i_wiki, doc in enumerate(open(args.wiki_file)):
    if i_wiki < wiki_split[0]:
        key = 'train'
    elif i_wiki < (wiki_split[0] + wiki_split[1]):
        key = 'val'
    else:
        key = 'test'
    for s in re.findall(r'(\\\?\\Q\d+)', doc):
        wd_ent = s.replace('\\?\\', '')
        if kb_wd.in_kb(wd_ent):
            wiki_ents[key] += [kb_wd.wikidata_to_wikipedia[wd_ent]]
        else:
            not_in_kb_wd += [wd_ent]
print(f"{len(not_in_kb_wd)} entities not in current Wikidata KB (expect 0!)")

conll_split = [946, 216, 231]
conll_ents = defaultdict(list)
for i_conll, doc in enumerate(get_docs(args.conll_file)):
    if i_conll < conll_split[0]:
        key = 'train'
    elif i_conll < (conll_split[0] + conll_split[1]):
        key = 'val'
    else:
        key = 'test'
    for token in doc.tokens:
        ent = token.true_label
        if ent not in ['I', 'O', 'B']:
            conll_ents[key] += [ent.replace('_', ' ')]

preds = {}
for line in open(args.pred_file):
    vals = line.strip().split('\t')
    if len(vals) < 4:
        print(vals)
    preds[tuple(vals[:2])] = kb_wd.get_w2v_entity(vals[3])

labels = {}
for line in open(args.gt_file):
    vals = line.strip().split('\t')
    if len(vals) < 4:
        print(vals)
    labels[tuple(vals[:2])] = kb_wd.get_w2v_entity(vals[3])

wiki_ents_train = set(wiki_ents['train'])
conll_ents_train = set(conll_ents['train'])

# Label spans that are not in pred spans
md_false_neg_spans = \
    list(set(labels.keys()).difference(set(preds.keys())))
# Pred spans that are not in label spans
md_false_pos_spans = \
    list(set(preds.keys()).difference(set(labels.keys())))
# Spans in both labels and preds
md_true_pos_spans = \
    list(set(labels.keys()).intersection(set(preds.keys())))

md_false_neg = defaultdict(list)
md_false_pos = defaultdict(list)
md_true_pos = defaultdict(list)


def add_to_dict(d, ent, span):
    if ent in wiki_ents_train:
        d['in_w'] += [span]
    else:
        d['not_in_w'] += [span]
    if ent in conll_ents_train:
        d['in_c'] += [span]
    else:
        d['not_in_c'] += [span]


for span in md_false_neg_spans:
    add_to_dict(md_false_neg, labels[span], span)
for span in md_false_pos_spans:
    add_to_dict(md_false_pos, preds[span], span)
for span in md_true_pos_spans:
    add_to_dict(md_true_pos, labels[span], span)

# MD TP where predicted entity and label entity is the same
ed_true = defaultdict(list)
# MD TP where predicted entity and label entity are different
# These are considered both False Positive and False Negative
ed_false = defaultdict(list)
for span in md_true_pos_spans:
    if labels[span] == preds[span]:
        add_to_dict(ed_true, labels[span], span)
    else:
        add_to_dict(ed_false, labels[span], span)
# Include Fale Negative MD predictions to make EL rather than ED stats
# for span in md_false_neg_spans:
#     add_to_dict(ed_false, labels[span], span)

print(
    f'In-KB Mention Detection:\n'
    f'- {len(md_true_pos["in_w"] + md_true_pos["not_in_w"]):>4} TP\n'
    f'->  {len(md_true_pos["in_w"]):>4} In WA Train\n'
    f'->  {len(md_true_pos["not_in_w"]):>4} Not in WA Train\n'
    f'- {len(md_false_neg["in_w"] + md_false_neg["not_in_w"]):>4} FN\n'
    f'->  {len(md_false_neg["in_w"]):>4} In WA Train\n'
    f'->  {len(md_false_neg["not_in_w"]):>4} Not in WA Train\n'
    f'- {len(md_false_pos["in_w"] + md_false_pos["not_in_w"]):>4} FP\n'
    f'->  {len(md_false_pos["in_w"]):>4} In WA Train\n'
    f'->  {len(md_false_pos["not_in_w"]):>4} Not in WA Train'
)
print(
    f'In-KB Entity Linking:\n'
    f'- {len(ed_true["in_w"] + ed_true["not_in_w"]):>4} TP\n'
    f'->  {len(ed_true["in_w"]):>4} In WA Train\n'
    f'->  {len(ed_true["not_in_w"]):>4} Not in WA Train\n'
    f'- {len(ed_false["in_w"] + ed_false["not_in_w"]):>4} False\n'
    f'->  {len(ed_false["in_w"]):>4} In WA Train\n'
    f'->  {len(ed_false["not_in_w"]):>4} Not in WA Train\n'
)

print(
    f'In-KB Mention Detection:\n'
    f'- {len(md_true_pos["in_c"] + md_true_pos["not_in_c"]):>4} TP\n'
    f'->  {len(md_true_pos["in_c"]):>4} In CoNLL Train\n'
    f'->  {len(md_true_pos["not_in_c"]):>4} Not in CoNLL Train\n'
    f'- {len(md_false_neg["in_c"] + md_false_neg["not_in_c"]):>4} FN\n'
    f'->  {len(md_false_neg["in_c"]):>4} In CoNLL Train\n'
    f'->  {len(md_false_neg["not_in_c"]):>4} Not in CoNLL Train\n'
    f'- {len(md_false_pos["in_c"] + md_false_pos["not_in_c"]):>4} FP\n'
    f'->  {len(md_false_pos["in_c"]):>4} In CoNLL Train\n'
    f'->  {len(md_false_pos["not_in_c"]):>4} Not in CoNLL Train'
)
print(
    f'In-KB Entity Linking:\n'
    f'- {len(ed_true["in_c"] + ed_true["not_in_c"]):>4} TP\n'
    f'->  {len(ed_true["in_c"]):>4} In CoNLL Train\n'
    f'->  {len(ed_true["not_in_c"]):>4} Not in CoNLL Train\n'
    f'- {len(ed_false["in_c"] + ed_false["not_in_c"]):>4} False\n'
    f'->  {len(ed_false["in_c"]):>4} In CoNLL Train\n'
    f'->  {len(ed_false["not_in_c"]):>4} Not in CoNLL Train\n'
)

t_in_w_in_c = len(
        set(ed_true["in_w"]).intersection(set(ed_true["in_c"]))
    )
t_not_in_w_in_c = len(
        set(ed_true["not_in_w"]).intersection(set(ed_true["in_c"]))
    )
f_in_w_in_c = len(
        set(ed_false["in_w"]).intersection(set(ed_false["in_c"]))
    )
f_not_in_w_in_c = len(
        set(ed_false["not_in_w"]).intersection(set(ed_false["in_c"]))
    )
t_in_w_not_in_c = len(
        set(ed_true["in_w"]).intersection(set(ed_true["not_in_c"]))
    )
t_not_in_w_not_in_c = len(
        set(ed_true["not_in_w"]).intersection(set(ed_true["not_in_c"]))
    )
f_in_w_not_in_c = len(
        set(ed_false["in_w"]).intersection(set(ed_false["not_in_c"]))
    )
f_not_in_w_not_in_c = len(
        set(ed_false["not_in_w"]).intersection(set(ed_false["not_in_c"]))
    )

print("\nEntity Disambiguation predictions"
      "\n(i.e. where Mention Detection is correct)"
      "\nC = AIDA-CoNLL Train; WA = Wikipedia Articles Train"
      "\nCorrect & Wrong Disambiguation:\n")

print("{:^8} | {:^8}  {:^8}".format("", "WA", "Not WA"))
print('-'*30)
print("{:^8} | {:^8}  {:^8}".format(
    "",
    t_in_w_in_c,
    t_not_in_w_in_c,
))
print("{:^8} | {:^8}  {:^8}".format(
        "C",
        "& " + str(f_in_w_in_c),
        "& " + str(f_not_in_w_in_c)
))
print("{:^8} | {:^8}  {:^8}".format(
    "",
    '(= ' + str(t_in_w_in_c + f_in_w_in_c) + ')',
    '(= ' + str(t_not_in_w_in_c + f_not_in_w_in_c) + ')',
))

print()

print("{:^8} | {:^8}  {:^8}".format(
    "",
    t_in_w_not_in_c,
    t_not_in_w_not_in_c,
))
print("{:^8} | {:^8}  {:^8}".format(
        "Not C",
        "& " + str(f_in_w_not_in_c),
        "& " + str(f_not_in_w_not_in_c),
))
print("{:^8} | {:^8}  {:^8}".format(
    "",
    '(= ' + str(t_in_w_not_in_c + f_in_w_not_in_c) + ')',
    '(= ' + str(t_not_in_w_not_in_c + f_not_in_w_not_in_c) + ')',
))

# popularity_correlation.py

from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia
from src.toolbox import get_docs
from collections import defaultdict, Counter
from scipy.stats.stats import pearsonr
import argparse


default_gt = '/models/trained/20211129_1004/saved_20211129_1705/' \
    + 'wikipedia2vec/conll_test_cs_oo_kb_ground_truth.tsv'
default_pred = '/models/trained/20211129_1004/saved_20211129_1705/' \
    + 'wikipedia2vec/conll_test_cs_oo_kb_predictions.tsv'

parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "--pred_file", type=str, default=default_pred,
        help="path to a predictions .tsv file, primarily for AIDA-CoNLL Test"
    )
parser.add_argument(
        "--gt_file", type=str, default=default_gt,
        help="path to a ground truth .tsv file, primarily for AIDA-CoNLL Test"
    )
args = parser.parse_args()

pred_file = default_pred  # args.pred_file
gt_file = default_gt  # args.gt_file

kb = KnowledgeBaseWikipedia(
    '/ex_data/knowledgebases/wikipedia2vec/enwiki_20180420_100d.pkl',
    '/ex_data/alias_dict.json',
    '/ex_data/entity_dict.json')

conll_split = [946, 216, 231]
conll_ents = defaultdict(list)
for i_conll, doc in enumerate(get_docs(
            '/ex_data/annotated_datasets/conll-wikipedia-iob-annotations'
        )):
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

conll_ents['test_unseen'] = \
    [e for e in conll_ents['test'] if e not in conll_ents['train']]
conll_ents['test_seen'] = \
    [e for e in conll_ents['test'] if e in conll_ents['train']]
conll_ents['val_unseen'] = \
    [e for e in conll_ents['val'] if e not in conll_ents['train']]
conll_ents['val_seen'] = \
    [e for e in conll_ents['val'] if e in conll_ents['train']]
conll_ents['all'] = \
    conll_ents['train'] + conll_ents['test'] + conll_ents['val']
conll_ents['testval_unseen'] = \
    conll_ents['test_unseen'] + conll_ents['val_unseen']
conll_ents['testval_seen'] = \
    conll_ents['test_seen'] + conll_ents['val_seen']

# preds = {}
# for line in open(pred_file):
#     vals = line.strip().split('\t')
#     if len(vals) < 4:
#         print(vals)
#     preds[tuple(vals[:2])] = kb.get_w2v_entity(vals[2])

# labels = {}
# for line in open(gt_file):
#     vals = line.strip().split('\t')
#     if len(vals) < 4:
#         print(vals)
#     labels[tuple(vals[:2])] = kb.get_w2v_entity(vals[2].replace('_', ' '))

# md_true_positive_spans = \
#     list(set(labels.keys()).intersection(set(preds.keys())))
# md_false_negative_spans = \
#     list(set(labels.keys()).difference(set(preds.keys())))

# tp_spans_seen = [
#         span for span in md_true_positive_spans
#         if labels[span] in conll_ents['train']
#     ]
# tp_spans_unseen = [
#         span for span in md_true_positive_spans
#         if labels[span] not in conll_ents['train']
#     ]
# fn_spans_seen = [
#         span for span in md_false_negative_spans
#         if labels[span] in conll_ents['train']
#     ]
# fn_spans_unseen = [
#         span for span in md_false_negative_spans
#         if labels[span] not in conll_ents['train']
#     ]

scores = {}
for k in conll_ents:
    cnts = sorted(
            list(Counter(conll_ents[k]).items()),
            key=lambda x: x[1],
            reverse=True
        )
    scores[k] = []
    for ent in cnts:
        if kb.get_w2v_entity(ent[0]):
            score = kb.wikipedia2vec.get_entity(ent[0]).count
            scores[k] += [(ent[0], ent[1], score)]
    print()
    print(f"Dataset {k:<5}: {len(scores[k])} / {len(cnts)} entities in KB")
    freq = [e[1] for e in scores[k]]
    pop = [e[2] for e in scores[k]]
    corr, pr = pearsonr(freq, pop)
    print("   Correlation between entity frequency in dataset "
          "and Wikipedia popularity")
    print(f"     pearson correlation: {corr:.4f}, p-value: {pr:.4E}")

# Correlation between frequency in Train and in Test of 'seen' entities
cnt_train = Counter(conll_ents['train'])
pop_seen = [
        kb.wikipedia2vec.get_entity(e[0]).count for e in scores['test_seen']
        if kb.get_w2v_entity(e[0])
    ]
freq_seen_train = [cnt_train[e[0]] for e in scores['test_seen']]
corr, pr = pearsonr(
        pop_seen,
        freq_seen_train,
    )
print()
print("Seen Test entities:")
print("   Correlation between !!!popularity and frequency of SEEN "
      "entities in Train!!!:")
print(f"     pearson correlation: {corr:.4f}, p-value: {pr:.4E}")

# Correlation between frequency in Train and in Test of 'unseen' entities
cnt_val = Counter(conll_ents['val_unseen'])
cnt_test = Counter(conll_ents['test_unseen'])
pop_unseen = [
        kb.wikipedia2vec.get_entity(e[0]).count for e in scores['test_unseen']
        if kb.get_w2v_entity(e[0])
    ]
freq_unseen_testval = []
for e in scores['test_unseen']:
    score = e[1]
    score += cnt_val[e[0]] if e[0] in cnt_val else 0
    freq_unseen_testval += [score]
corr, pr = pearsonr(
        pop_unseen,
        freq_unseen_testval,
    )
print()
print("Unseen Test entities:")
print("   Correlation between !!!popularity and frequency of UNSEEN "
      "entities in Test and Val!!!:")
print(f"     pearson correlation: {corr:.4f}, p-value: {pr:.4E}")

conll_file = '/ex_data/annotated_datasets/conll-wikipedia-iob-annotations'
conll_mentions = defaultdict(list)
for i_doc, doc in enumerate(get_docs(conll_file)):
    conll_dataset = 'None'
    if i_doc < 946:
        conll_dataset = 'Train'
    elif i_doc < 1162:
        conll_dataset = 'Val'
    elif i_doc < 1393:
        conll_dataset = 'Test'
    # In a labeled mention ?
    m = False
    for tok in doc.tokens:
        # Add labeled mentions to respective dataset list
        if tok.true_label not in ['I', 'O', 'B']:
            conll_mentions[conll_dataset] += [(tok.true_label, [tok.text])]
            conll_mentions['All'] += [(tok.true_label, [tok.text])]
            m = True
        # Append 'I' text if they follow a labelled mention
        elif tok.true_label == 'I' and m:
            tup = conll_mentions[conll_dataset][-1]
            tup = (tup[0], tup[1] + [tok.text])
            conll_mentions[conll_dataset][-1] = tup
            conll_mentions['All'][-1] = tup
        # If token is 'O' or 'B' or 'I' not following labeled token
        else:
            m = False

for dataset, mentions in conll_mentions.items():
    # Iterate the mention tuples
    for i, mention in enumerate(mentions):
        mention_id = mention[0].replace('_', ' ')
        mention_text = ' '.join(mention[1])
        candidate_set = []
        if kb.in_cand_gen(mention_id):
            candidate_set = sorted(
                    kb.get_candidate_set_from_mention(mention_text),
                    key=lambda x: kb.wikipedia2vec.get_entity(x).count,
                    reverse=True,
                )
        conll_mentions[dataset][i] = (mention_id, mention_text, candidate_set)

conll_mentions['Test_seen'] = \
    [m for m in conll_mentions['Test']
     if m[0] in conll_ents['train'] and m[0] in m[2]]
conll_mentions['Test_unseen'] = \
    [m for m in conll_mentions['Test']
     if m[0] not in conll_ents['train'] and m[0] in m[2]]

seen_label_pos = [m[2].index(m[0]) for m in conll_mentions['Test_seen']]
seen_avg_label_pos = sum(seen_label_pos)/len(seen_label_pos)
unseen_label_pos = [m[2].index(m[0]) for m in conll_mentions['Test_unseen']]
unseen_avg_label_pos = sum(unseen_label_pos)/len(unseen_label_pos)

print("For {} Seen entities: ".format(
    len(conll_mentions['Test_seen'])
), end='')
print("GT Label's avg position in Candidate Set: {:.4f} (median: {})".format(
    seen_avg_label_pos, seen_label_pos[int(len(seen_label_pos)/2)]
))
print("For {} Unseen entities: ".format(
    len(conll_mentions['Test_unseen'])
), end='')
print("GT Label's avg position in Candidate Set: {:.4f} (median: {})".format(
    unseen_avg_label_pos, unseen_label_pos[int(len(unseen_label_pos)/2)]
))

print()

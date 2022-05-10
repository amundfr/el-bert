"""
Script to evaluate KB for Candidate Generation

Run as a module to avoid path and import problems:

    python -m scripts.evaluate_cg -h

"""

import configparser
import argparse
from collections import Counter, defaultdict
from src.toolbox import get_docs, get_knowledge_base

parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "-p", "--plot", action='store_true', default=False,
        help="flag: if provided, saves plots of candidate set lengths"
             " and alias lengths"
    )
parser.add_argument(
        "--kb_type", type=str, nargs=1, default='pedia',
        choices=['pedia', 'data'],
        help="which Knowledge Base TYPE to use. "
    )
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

kb = get_knowledge_base(config, args.kb_type[0], with_cg=True)

conll_file = config['DATA']['Annotated Dataset']
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

# Entity has at least one alias in Candidate Generation
in_cg = defaultdict(list)
# ... and in Wikipedia2vec
in_cg_in_kb = defaultdict(list)
# ... and NOT in Wikipedia2vec
in_cg_not_in_kb = defaultdict(list)
# ... and in Wikipedia2vec and the Candidate set for its mention text
in_cg_in_kb_in_cs = defaultdict(list)
# ... alias count for those entities
in_cg_in_kb_in_cs_aliases = defaultdict(list)
# ... and in Wikipedia2vec and NOT the Candidate set for its mention text
in_cg_in_kb_not_in_cs = defaultdict(list)
# ... alias count for those entities
in_cg_in_kb_not_in_cs_aliases = defaultdict(list)
# ... and NOT in Wikipedia2vec and in the Candidate set for its mention text
in_cg_not_in_kb_in_cs = defaultdict(list)
# ... and NOT in Wikipedia2vec and NOT the Candidate set for its mention text
in_cg_not_in_kb_not_in_cs = defaultdict(list)
# Entity has no aliases in Candidate Generation (unfindable by CG)
not_in_cg = defaultdict(list)
# ... and is in Wikipedia2vec
not_in_cg_in_kb = defaultdict(list)
# ... and is NOT in Wikipedia2vec
not_in_cg_not_in_kb = defaultdict(list)
# Mention text gave no candidates (i.e. "alias not in CG")
in_kb_empty_cs = defaultdict(list)
# Mention text gave no candidates (i.e. "alias not in CG")
not_in_kb_empty_cs = defaultdict(list)
# Length of candidate set, when ground truth entity in candidate set
in_cs_len_cs = defaultdict(list)
# Length of candidate set, when ground truth entity NOT in candidate set
not_in_cs_len_cs = defaultdict(list)

# iterate the dataset and list of mentions in each dataset
for dataset, mentions in conll_mentions.items():
    emtis = 0
    # Iterate the mention tuples
    for mention in mentions:
        mention_id = mention[0]
        mention_text = ' '.join(mention[1])
        # In Candidate Generation system
        if kb.in_cand_gen(mention_id):
            in_cg[dataset] += [mention]
            candidate_set = kb.get_candidate_set_from_mention(mention_text)
            if len(candidate_set) == 0:
                emtis += 1
            # In Wiki KB
            kb_entity = kb.get_kb_entity(mention_id)
            if kb_entity:
                in_cg_in_kb[dataset] += [mention]
                if kb_entity in candidate_set:
                    in_cg_in_kb_in_cs[dataset] += [mention]
                    in_cg_in_kb_in_cs_aliases[dataset] += \
                        [len(kb.entity_dict[mention_id])]
                    in_cs_len_cs[dataset] += [len(candidate_set)]
                else:
                    in_cg_in_kb_not_in_cs[dataset] += [mention]
                    in_cg_in_kb_not_in_cs_aliases[dataset] += \
                        [len(kb.entity_dict[mention_id])]
                    if len(candidate_set) == 0:
                        in_kb_empty_cs[dataset] += [mention]
                    else:
                        not_in_cs_len_cs[dataset] += [len(candidate_set)]
            # Not in Wiki KB
            else:
                if mention_id in candidate_set:
                    in_cg_not_in_kb_in_cs[dataset] += [mention]
                else:
                    in_cg_not_in_kb_not_in_cs[dataset] += [mention]
                    if len(candidate_set) == 0:
                        not_in_kb_empty_cs[dataset] += [mention]
                in_cg_not_in_kb[dataset] += [mention]
        # Entity not in Candidate Generation system
        else:
            not_in_cg[dataset] += [mention]
            # In Wiki KB
            if not kb.in_kb(mention_id):
                not_in_cg_not_in_kb[dataset] += [mention]
            else:
                not_in_cg_in_kb[dataset] += [mention]
    # print(f'\n Dataset: {dataset}, emptis: {emtis}')

# print(" {:>10,} entities in Wikipedia2vec Knowledge Base"
#       .format(len(kb.wikidata_to_wikipedia)))
print(" {:>10,} entities in Candidate Generation (CG) system"
      .format(kb.n_cg_entities()))
print(" {:>10,} aliases in Candidate Generation (CG) system"
      .format(kb.n_cg_aliases()))


def do_count(li):
    mentions = len(li)
    unique_entities = len(set([m[0] for m in li]))
    return mentions, unique_entities


dataset_order = ['All', 'Test', 'Val', 'Train']
for dataset in dataset_order:
    print(f"\n\n===  Dataset AIDA-CoNLL {dataset}  ===")

    print("Gold recall on mentions and entities:")
    cg_success = do_count(in_cg_in_kb_in_cs[dataset])
    total = do_count(not_in_cg[dataset] + in_cg[dataset])
    not_in_kb = do_count(
            not_in_cg_not_in_kb[dataset] + in_cg_not_in_kb[dataset]
        )
    print(f"{100*cg_success[0]/(total[0]-not_in_kb[0]):.2f} /"
          f" {100*cg_success[1]/(total[1]-not_in_kb[1]):.2f}")

    print("mentions / entities in CoNLL...")
    print("\n--  Total in CoNLL:  --")
    print("  {:>6} / {:>5}  in CoNLL".format(
            *do_count(not_in_cg[dataset] + in_cg[dataset])))

    print("\n--  Entity not in CG "
          "(Ground Truth entity has no aliases in CG system)  --")
    print("  {:>6} / {:>5}  ".format(
                *do_count(not_in_cg_in_kb[dataset])
            ) + "not in CG, but in Wiki KB")
    print(" +{:>6} / {:>5}  ".format(
                *do_count(not_in_cg_not_in_kb[dataset])
            ) + "not in CG and not in Wiki KB")
    print(" ={:>6} / {:>5}  ".format(
                *do_count(not_in_cg[dataset])
            ) + "not in CG")

    print("\n--  Entity in CG (Ground Truth entity has at least one alias"
          " in the CG system)  --")
    print("  {:>6} / {:>5}  ".format(
                *do_count(in_cg_not_in_kb[dataset])
            ) + "in CG but not in Wiki KB")
    print(" ={:>6} / {:>5}  ".format(
                *do_count(in_cg[dataset])
            ) + "in CG")

    if len(in_cg_not_in_kb[dataset]) > 0:
        print("\n--  Entity in CG, not in Wikipedia2vec KB  --")
        print("   {:>6} / {:>5}  ".format(
                    *do_count(in_cg_not_in_kb_in_cs[dataset])
                ) + "in CG, not in Wiki KB, but is in Candidate Set")
        print(" + {:>6} / {:>5}  ".format(
                    *do_count(in_cg_not_in_kb_not_in_cs[dataset])
                ) + "in CG, not in Wiki KB, and not in Candidate Set")
        print(" ( {:>6} / {:>5}  ".format(
                    *do_count(not_in_kb_empty_cs[dataset])
                ) + "... of which Candidate Set was empty )")
        print(" = {:>6} / {:>5}  ".format(
                *do_count(in_cg_not_in_kb[dataset])
            ) + "in CG but not in Wiki KB")

    print("\n--  Entity in CG, and in Wikipedia2vec KB  --")
    print("  {:>6} / {:>5}  ".format(
                *do_count(in_cg_in_kb_in_cs[dataset])
            ) + "in CG, in Wiki KB, and in Candidate Set (CG SUCCESS!)")
    if len(in_cs_len_cs[dataset]) > 0:
        print(" ({:>5.1f} ".format(
                sum(in_cs_len_cs[dataset]) / len(in_cs_len_cs[dataset])
            ) +
            " average number of candidates when entity in candidate set)")
        print(" ({:>5.0f} ".format(
                sum(1 for cs_len in in_cs_len_cs[dataset] if cs_len == 1)
            ) +
            " only one candidate (the correct one))")
        print(" ({:>5.0f} ".format(
                sum(in_cg_in_kb_in_cs_aliases[dataset])
                / len(in_cg_in_kb_in_cs_aliases[dataset])
            ) +
            " average number of aliases for entities that were a success)")
    print(" + {:>6} / {:>5}  ".format(
                *do_count(in_cg_in_kb_not_in_cs[dataset])
            ) + "in CG, in Wiki KB, and not in Candidate Set")
    if len(not_in_cs_len_cs[dataset]) > 0:
        # non_zero_lengths = \
        #     [length for length in not_in_cs_len_cs[dataset] if length != 0]
        print(" ({:>5.1f} ".format(
                sum(not_in_cs_len_cs[dataset]) / len(not_in_cs_len_cs[dataset])
            ) +
            " average number of candidates when entity not in candidate set)")
        print(" ({:>5.0f} ".format(
                sum(in_cg_in_kb_not_in_cs_aliases[dataset])
                / len(in_cg_in_kb_not_in_cs_aliases[dataset])
            ) +
            " average number of aliases for entities that were not a success)")
    print(" ( {:>6} / {:>5}  ".format(
                *do_count(in_kb_empty_cs[dataset])
            ) + "... of which Candidate Set was empty)")
    print(" = {:>6} / {:>5}  ".format(
                *do_count(in_cg_in_kb[dataset])
            ) + "in CG and Wiki KB")

    print("\n--  Entity in Wikipedia2vec KB, but some CG error  --")
    print("  {:>6} / {:>5}  ".format(
                *do_count(not_in_cg_in_kb[dataset])
            ) + "not in CG, but in Wiki KB")
    print(" +{:>6} / {:>5}  ".format(
                *do_count(in_cg_in_kb_not_in_cs[dataset])
            ) + "in CG, in Wiki KB, but not in Candidate Set")
    print(" ( {:>6} / {:>5}  ".format(
                *do_count(in_kb_empty_cs[dataset])
            ) + "... of which Candidate Set was empty)")
    print(" ={:>6} / {:>5}  ".format(
                *do_count(
                        not_in_cg_in_kb[dataset]
                        + in_cg_in_kb_not_in_cs[dataset]
                    ),
            ) + "errors from CG alone")
print()

if args.plot:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('pdf')
    mpl.rc('text', usetex=True)
    mpl.rc('font', **{'family': "sans-serif"})
    params = {'text.latex.preamble': r'\usepackage{amsmath}'}

    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    plt.rcParams.update(params)

    # width as measured in inkscape
    width = 2*3.487
    height = width / 1.618

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.95)
    dataset = in_cs_len_cs['Val']
    max_len = max(dataset)

    # plt.hist(dataset, bins=range(0, max_len, 25), rwidth=0.8)
    cnt = Counter(dataset)
    gap = 10
    bins = list(range(int(gap/2), 50*(1+int(max_len/50)), gap))
    height = [0] * len(bins)
    for cs_len, count in cnt.items():
        height[int(cs_len/gap)] += count

    plt.bar(bins, height=height, width=0.9*gap)
    # plt.scatter(cnt.keys(), cnt.values(), marker='.', s=10)
    # xy = sorted(cnt.items(), key=lambda x: x[0])
    # plt.plot([xy_[0] for xy_ in xy], [xy_[1] for xy_ in xy])

    ax.set_xticks(range(0, 50*(1+int(max_len/50)), 50))
    for i, tick in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            tick.set_visible(False)
    # ax.set_yscale('log')
    ax.set_ylabel('Mentions')
    ax.set_xlabel('Length of Candidate Set')
    # ax.set_xlim(0, 3*np.pi)

    # fig.set_size_inches(width, height)
    fig.savefig('cs_length_bar_in_cs_val.pdf')

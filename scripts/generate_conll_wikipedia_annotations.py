"""
Script produces a Wikipedia-annotated CoNLL document file.
This script only needs to be run once. The default out-file path is
/ex_data/annotated_datasets/conll-wikipedia-iob-annotations

The resulting annotated document can be used in generate_input_data.py

Run as module:
    python -m scripts.generate_conll_wikipedia_annotations -h

"""
import argparse
import json
import os.path
from configparser import ConfigParser
from wikipedia2vec import Wikipedia2Vec
from urllib.parse import unquote
from unidecode import unidecode
from src.toolbox import get_docs


config = ConfigParser()
config.read('config.ini')
w2v_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']
w2v_file = config['KNOWLEDGE BASE']['Wikipedia2vec File']
w2v_path = os.path.join(w2v_dir, w2v_file)

parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "--wikipedia2vec", type=str,
        default=w2v_path,
        help="path to Wikipedia2vec .pkl file"
    )
parser.add_argument(
        "--conll_wikidata_file", type=str,
        default='/ex_data/annotated_datasets/conll-wikidata-iob-annotations',
        help="path to Wikidata-annotated AIDA-CoNLL file"
    )
parser.add_argument(
        "--wikipedia_labels", type=str,
        default='/ex_data/AIDA-YAGO2-annotations.tsv',
        help="path to .tsv file with Wikipedia annotations" +
             " for AIDA-CoNLL dataset"
    )
parser.add_argument(
        "--wikidata_wikipedia_mapping", type=str,
        default='/ex_data/wikidata_wikipedia_mapping.csv',
        help="path to .csv file with mapping from Wikidata-to-Wikipedia " +
             "entity URLs"
    )
parser.add_argument(
        "out_file", type=str,
        default='/ex_data/annotated_datasets/conll-wikipedia-iob-annotations',
        help="path to output file for Wikipedia-annotated AIDA-CoNLL dataset"
    )
args = parser.parse_args()

wikipedia_labels_file = args.wikipedia_labels
conll_file = args.conll_wikidata_file
out_file = args.out_file
wikipedia2vec_file = args.wikipedia2vec
wikidata_to_wikipedia_file = args.wikidata_wikipedia_mapping
wikipedia_url_prefix = 'https://en.wikipedia.org/wiki/'
wikidata_url_prefix = 'http://www.wikidata.org/entity/'

labels = []

for line in open(wikipedia_labels_file):
    vals = line.strip().split('\t')
    if len(vals) == 1:
        if vals[0].startswith('-DOCSTART-'):
            labels += [[]]
    elif len(vals) == 2:
        labels[-1] += [tuple([v.strip() for v in vals])]
    elif len(vals) in [4, 5]:
        entity = vals[1].encode().decode("unicode-escape")
        labels[-1] += [tuple([int(vals[0]), entity])]

w2v = Wikipedia2Vec.load(wikipedia2vec_file)
wikidata_to_wikipedia_mapping = []
wikidata_to_wikipedia = {}
wikipedia_to_wikidata = {}
duplicates = []
for i, line in enumerate(open(wikidata_to_wikipedia_file)):
    if (i % 100000) == 0:
        print(f"  {i} mapping entities processed", end='\r')
    entity_urls = line[1:-2].split('>,<')
    wikipedia_entity = entity_urls[0][len(wikipedia_url_prefix):]
    wikipedia_entity = unquote(wikipedia_entity)
    wikidata_entity = entity_urls[1][len(wikidata_url_prefix):]

    # Avoid duplicates
    if wikipedia_entity in wikipedia_to_wikidata:
        duplicates += [wikipedia_entity, wikidata_entity]
        continue

    wikipedia_to_wikidata[wikipedia_entity] = wikidata_entity
    wikidata_to_wikipedia[wikidata_entity] = wikipedia_entity


useful_mappings = {}
no_need_for_mapping = []
mapping_does_not_help = []
not_in_mapping = []
with open(out_file, 'w') as f_out:
    for i_doc, doc in enumerate(get_docs(conll_file)):
        for doc_label in labels[i_doc]:
            if doc_label[1] == '--NME--':
                continue
            wikipedia_id = doc_label[1]
            # First, look for entity with provided label
            entity = w2v.get_entity(unidecode(wikipedia_id).replace('_', ' '))
            # If it does not exist:
            if not entity:
                # Get the current Wikidata annotation of the token
                wikidata_id = doc.tokens[int(doc_label[0])].true_label
                if not wikidata_id == 'B' \
                        and wikidata_id in wikidata_to_wikipedia:
                    # Get Wikipedia entity from mapping
                    entity = w2v.get_entity(
                                unidecode(
                                        wikidata_to_wikipedia[wikidata_id]
                                    ).replace('_', ' ')
                            )
                    if entity:
                        wikipedia_id = \
                            unidecode(wikidata_to_wikipedia[wikidata_id])
                        useful_mappings[unidecode(doc_label[1])] = wikipedia_id
                    else:
                        mapping_does_not_help += [wikipedia_id]
                else:
                    not_in_mapping += [wikipedia_id]
            else:
                wikipedia_id = entity.title.replace(' ', '_')  # doc_label[1]
                no_need_for_mapping += [wikipedia_id]
            # A final check (Wikipedia2vec redirect can lead to "dead-ends")
            if not wikipedia_id == doc_label[1] \
                    and not w2v.get_entity(wikipedia_id.replace('_', ' ')):
                print(f'Dead end: {wikipedia_id}, ' +
                      f'reverting to: {doc_label[1]}')
                wikipedia_id = doc_label[1]
            doc.tokens[int(doc_label[0])].true_label = wikipedia_id
        f_out.write(f'{i_doc}\t{doc.get_truth()}\n')

print(
    f"{len(no_need_for_mapping)} Mentions were in Wikpedia2vec without mapping"
)
print(
    f"{len(useful_mappings)} Mentions were in Wikpedia2vec after using " +
    "Wikidata to Wikipedia mapping"
)
print(
    f"{len(useful_mappings)+len(no_need_for_mapping)} Mentions total in " +
    " Wikpedia2vec"
)
print(
    f"{len(mapping_does_not_help)} Mentions were not in Wikipedia2vec, " +
    "even with Mapping"
)
print(
    f"{len(not_in_mapping)} Mentions were not in Wikipedia2vec or Mapping"
)
print(
    f"{len(mapping_does_not_help+not_in_mapping)} Mentions total not in " +
    "Wikipedia2vec"
)

useful_mappings_file = '/ex_data/extra_redirect.json'
json.dump(useful_mappings, open(useful_mappings_file, 'w'))

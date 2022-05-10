"""
With a .json file with Wikipedia articles with keys
"id", "title", "text", "links", "url", "evaluation_span"
and a KnowledgeBase object, this file creates a dataset with training articles
where at least two entities are in the KnowledgeBase. The dataset is written
to file in the same format as the Wikidata annotated CoNLL dataset.

Run as a module to avoid import problems:

python -m scripts.make_training_data_from_wikipedia_articles -h

"""

from src.toolbox import get_knowledge_base
import json
import argparse
import re
import configparser


parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
parser.add_argument(
        "--training_articles", type=str,
        default='/w_data/training_articles.jsonl',
        help="path to .jsonl file with training articles"
    )
dest_file = '/ex_data/annotated_datasets/train_articles_dataset_score_0.txt'
parser.add_argument(
        "--destination", type=str,
        default=dest_file,
        help="path to .jsonl file with training articles"
    )
args = parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')

kb = get_knowledge_base(config, 'data', with_cg=False)

f_in = open(args.training_articles)
f_out = open(args.destination, 'w')
n = 0
for i_line, line in enumerate(f_in):
    if i_line % 100000 == 0:
        print(
            f"Progress: {i_line:8>} documents processed ({n} added)", end='\r')
    # Read line as json
    doc = json.loads(line)

    # Find label entities that are in the KB
    ents = doc["links"]
    in_kb_ents = [tup for tup in ents if kb.in_kb(tup[1])]
    n_in_kb_ents = len(in_kb_ents)
    n_words = len(doc["text"].split())
    # If at least one entity is in KB, and they are not too infrequent.
    # For example 32 for CoNLL only-KB
    #  and 16 for other KBs
    if n_in_kb_ents > 1 and (n_words / n_in_kb_ents) < 16:
        n += 1
        annotated_text = doc["text"]
        # Start with the mentions
        # Annotate from the end to the start to keep spans relevant
        ents.sort(key=lambda tup: tup[0][0], reverse=True)
        for ent in ents:
            # Get the words of the mention text
            mention_text = \
                annotated_text[ent[0][0]:ent[0][1]].split(' ')
            # Annotate beginning with QID or 'B' if not in KB
            if ent in in_kb_ents:
                mention_text[0] += '\\?\\' + ent[1]
            else:
                mention_text[0] += '\\?\\B'
            # Annotate the rest with 'I'
            for i_word in range(1, len(mention_text)):
                mention_text[i_word] = mention_text[i_word] + '\\?\\I'
            # if something directly precedes the mention
            if ent[0][0] != 0 \
                    and annotated_text[ent[0][0] - 1] != ' ':
                # Add a space at the beginning
                mention_text[0] = ' ' + mention_text[0]
            # if something follows immediately after the mention
            if ent[0][1] != len(doc["text"]) \
                    and annotated_text[ent[0][1]] != ' ':
                # Add a space
                mention_text[-1] = mention_text[-1] + ' '
                # if n == 49275:
                #     print(annotated_text)

            # Place the annotated mention in the text
            annotated_text = annotated_text[:ent[0][0]] + \
                ' '.join(mention_text) + annotated_text[ent[0][1]:]

        # re.findall(r'(\w+\\\?\\Q\d+|\w+\\\?\\I)', annotated_text)
        doc_tokens = []
        words = annotated_text.split()
        for i_word, word in enumerate(words):
            # If already annotated
            if '\\?\\' in word:
                doc_tokens += [word]
            else:
                # Split into words and punctuations
                word_tokens = re.findall(r'\w+|[^\s\w]', word)
                word_tokens = \
                    [word_token + '\\?\\O' for word_token in word_tokens]
                doc_tokens += word_tokens
        f_out.write(str(n) + '\t' + ' '.join(doc_tokens) + '\n')

    # if n == 50000:
    #     break
print(f"\nGot a total of {n} documents\n")

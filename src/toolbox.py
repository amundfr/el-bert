import os.path
import datetime
import json
from configparser import ConfigParser
from src.conll_document import ConllDocument
from transformers import BertTokenizerFast
from src.knowledge_base_wikidata import KnowledgeBaseWikidata as KBWikidata
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia as KBWikipedia


def get_knowledge_base(
            config: ConfigParser,
            wiki: str = 'pedia',
            with_cg: bool = True
        ):
    """
    Get a KnowledgeBase object for the correct Wiki-type
    :param config: A loaded ConfigParser object with relevant paths
    :param wiki: 'pedia' or 'data' for resp. Wikipedia or Wikidata
    :param with_cg: 
        if True, loads KnowlegeBase for candidate generation
            (with full version of Wikipedia2vec)
        if False, loads KnowledgeBase with smaller Wikipedia2vec for faster
            similarity search
    :returns: a ready KnowledgeBaseWikidata or KnowledgeBaseWikipedia object
    """
    alias_file = ''
    entity_file = ''
    # If KB with Candidate Generation
    if with_cg:
        # Use full Wikipedia2vec
        kb_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']

        alias_file = config['KNOWLEDGE BASE']['Alias Dict']
        entity_file = config['KNOWLEDGE BASE']['Entity Dict']

    w2v_file = config['KNOWLEDGE BASE']['Wikipedia2Vec File']
    if wiki == 'pedia':
        if not with_cg:
            kb_dir = config['KNOWLEDGE BASE']['Compact Wikipedia KB']
        wiki2vec = os.path.join(kb_dir, w2v_file)
        return KBWikipedia(
                wikipedia2vec_file=wiki2vec,
                alias_dict_file=alias_file,
                entity_dict_file=entity_file,
            )
    elif wiki == 'data':
        if not with_cg:
            kb_dir = config['KNOWLEDGE BASE']['Compact Wikidata KB']
        wiki2vec = os.path.join(kb_dir, w2v_file)
        wiki_to_wiki_file = config['KNOWLEDGE BASE']['Wikidata To Wikipedia']
        wiki_to_wiki = os.path.join(kb_dir, wiki_to_wiki_file)
        return KBWikidata(
                wikipedia2vec_file=wiki2vec,
                wikidata_to_wikipedia_file=wiki_to_wiki,
                alias_dict_file=alias_file,
                entity_dict_file=entity_file,
            )
    else:
        raise ValueError(f"Expected wiki = 'pedia' or 'data'; got {wiki}")


def get_tokenizer(tokenizer_uri: str = 'bert-base-uncased'):
    """
    Get a tokenizer for a given BERT model
    :param tokenizer_uri: URI of a BERT tokenizer
        (same as corresponding BERT model)
    :returns: a BertTokenizerFast object
    """
    uncased = tokenizer_uri.endswith('uncased')
    return BertTokenizerFast.from_pretrained(
            tokenizer_uri,
            do_lower_case=uncased
        )


def get_docs(f: str = '/ex_data/conll-wikidata-iob-annotations'):
    """
    An iterator for documents in file
    :param f: a file with documents
    :yields: a document from the file
    """
    if not os.path.isfile(f):
        raise FileNotFoundError(
                f"Could not find annotated file at '{f}'."
            )
    else:
        with open(f) as file:
            for line in file:
                document = ConllDocument(line[:-1])
                yield document


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    :param elapsed: float seconds
    :returns: a formated string with 'd days, hh:mm:ss'
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as "d days, hh:mm:ss"
    return str(datetime.timedelta(seconds=elapsed_rounded))


def print_training_stats(training_stats):
    """
    Given a list of training stats, prints a table
    :param training_stats: training statistics from trainer.train
    """
    stats_h1 = "       Training            |    Validation"
    stats_h2 = " Epoch |   Time   |  Loss  |   Time   |  Loss  |" \
        " MD Mi.F1 | ED Mi.F1 "
    stats_fs = "{:^6} | {:>8} | {:6.4f} | {:>8} | {:6.4f} | " \
        "{:>7.4f}  | {:7.4f}"

    print("\n")
    print(stats_h1)
    print('_' * len(stats_h2))
    print(stats_h2)
    print('_' * len(stats_h2))

    for s in training_stats:
        print(stats_fs.format(
                s['epoch'], s['train time'], s['train loss'],
                s['val time'], s['val loss'],
                s['val MD F1'], s['val ED F1']
            ))


def print_training_stats_from_file(stats_file):
    """
    Given a .json file path with training stats,
    prints a table of training stats
    :param stats_file: file with training statistics from trainer.train
    """
    print_training_stats(load_stats_from_file(stats_file))


def write_stats_to_file(training_stats, directory):
    """
    Write the output training stats list from trainer.train to file
    :param training_stats: training statistics from trainer.train
    :param directory: destination directory for the resulting file
    """
    file_destination = os.path.join(directory, 'training_stats.json')
    with open(file_destination, 'w') as file:
        json.dump(training_stats, file)


def load_stats_from_file(stats_file):
    """
    :param stats_file: a .json file with training stats
    :returns: a list of training stats
    """
    return json.load(open(stats_file))

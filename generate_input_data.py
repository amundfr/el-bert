"""
Author: Amund Faller Råheim

Script to generate input data and write to file.
Necessary for train_model.py

Takes around 30 seconds for AIDA-CoNLL dataset.
"""
import argparse
from os import path
from configparser import ConfigParser
from src.knowledge_base_wikipedia import KnowledgeBaseWikipedia
from src.input_data_generator import InputDataGenerator
from src.dataset_generator import DatasetGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )
    parser.add_argument(
            "--input_file", type=str, default="",
            help="path to an annotated file with e.g. CoNLL data. " +
                 "Default is [DATA][Annotated Data] in config.ini"
        )
    parser.add_argument(
            "--output_dir", type=str, default="",
            help="path to save folder. " +
                 "Default is [DATA][Uncased Input Vectors Dir] in config.ini"
        )
    args = parser.parse_args()

    print("\nStarting input data generation."
          "\nThis script relies on paths in 'config.ini'")

    print("\n 1. Getting 'config.ini'")
    config = ConfigParser()
    config.read('config.ini')

    print("\n 2. Getting Knowledge Base")
    kb_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']
    print(f"Using Wikipedia2vec at {kb_dir}")

    w2vec_file = \
        path.join(kb_dir, config['KNOWLEDGE BASE']['Wikipedia2Vec File'])

    kb = KnowledgeBaseWikipedia(
            wikipedia2vec_file=w2vec_file,
        )

    tokenizer_id = config['MODEL']['Model ID']
    uncased = True if tokenizer_id.endswith('uncased') else False
    print(f"\n 3. Getting Input Data Generator for tokenizer '{tokenizer_id}'")
    print("Mode: {}cased".format("un" if uncased else ""))
    input_data_generator = InputDataGenerator(
            knowledgebase=kb,
            tokenizer_pretrained_id=tokenizer_id,
        )

    file = config['DATA']['Annotated Dataset']
    if args.input_file:
        file = args.input_file
    print(f"\n 4. Generating input vectors for annotated dataset at {file}")
    input_vectors = input_data_generator.generate_for_file(
            file=file,
            progress=True
        )

    if uncased:
        vector_dir = config['DATA']['Uncased Input Vectors Dir']
    else:
        vector_dir = config['DATA']['Cased Input Vectors Dir']
    if args.output_dir:
        vector_dir = args.output_dir
    print(f"\n 5. Writing data to directory '{vector_dir}'")

    dataset_generator = DatasetGenerator(*input_vectors)
    dataset_generator.save(vector_dir)

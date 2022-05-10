"""
Script to train a BERT model.
Assumes data from generate_input_data.py
"""
import transformers
import argparse
import time
from os import path
from configparser import ConfigParser
from src.dataset_generator import DatasetGenerator
from src.bert_model import BertMdEd, load_bert_from_file, save_model_to_dir
from src.trainer import ModelTrainer
from src.toolbox import get_knowledge_base, write_stats_to_file
import numpy as np
import torch
import random
seed_val = 142
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
transformers.logging.set_verbosity_error()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )

    parser.add_argument(
            "-n", "--new_model", action="store_true",
            help="get pretrained model from Huggingface, "
                 "and do not read model from disk"
        )
    parser.add_argument(
            "-g", "--gpu_index", type=int, nargs='?', default=0,
            help="the index of the GPU to use"
        )
    parser.add_argument(
            "-c", "--checkpoint", type=int, nargs='?', default=0,
            help="frequency of saved model checkpoints by epoch"
        )

    args = parser.parse_args()
    # print(f"args: {args}")

    print("\nStarting training script."
          "\nThis script relies on paths in 'config.ini',"
          "\n and requires the data from generate_input_data.py")

    print("\n 1. Getting 'config.ini'")
    config = ConfigParser()
    config.read('config.ini')

    model_dir = config['MODEL']['Bert Model Dir']
    loss_lambda = config.getfloat('TRAINING', 'Loss Lambda')
    hidden_output_layers = config.getboolean('MODEL', 'Hidden Output Layers')
    dropout_after_bert = config.getboolean('MODEL', 'Dropout After BERT')
    if args.new_model or not model_dir:
        bert_model_id = config['MODEL']['Model ID']
        print(f"\n 2. Getting BERT with model ID '{bert_model_id}'")
        model = BertMdEd.from_pretrained(
                bert_model_id,
                hidden_output_layers=hidden_output_layers,
                dropout_after_bert=dropout_after_bert,
                loss_lambda=loss_lambda,
            )
    else:
        print(f"\n 2. Loading BERT from directory '{model_dir}'")
        model = load_bert_from_file(
                model_dir,
                hidden_output_layers=hidden_output_layers,
                dropout_after_bert=dropout_after_bert,
                loss_lambda=loss_lambda,
            )
    print(f"{'U' if dropout_after_bert else 'Not u'}sing "
          f"dropout after BERT Embeddings")
    print(f"{'U' if hidden_output_layers else 'Not u'}sing "
          f"hidden output layers")
    print(f"Loss-lambda: {loss_lambda}")

    freeze_n = config.getint('TRAINING', 'Freeze N Transformers')
    if freeze_n:
        print(f"\n 2b. Freezeing {freeze_n} BERT tranformers")
        model.freeze_n_transformers(freeze_n)

    if model.config._name_or_path.endswith('uncased'):
        vectors_dir = config['DATA']['Uncased Input Vectors Dir']
    else:
        vectors_dir = config['DATA']['Cased Input Vectors Dir']

    print(f"\n 3. Loading dataset from '{vectors_dir}', and splitting data")
    docs_file = config['DATA']['Annotated Dataset']
    dataset_generator = DatasetGenerator.load(vectors_dir)
    batch_size = config.getint('TRAINING', 'Batch size')
    split = config['DATA']['Data Split'].split(', ')
    split = [int(s) for s in split]
    print(f"Using data split: {split}")
    data_loaders = dataset_generator.split_by_ratio(split, batch_size)

    epochs = config.getint('TRAINING', 'Epochs')
    print(f"\n 4. Getting model trainer for {epochs} training epochs")
    learning_rate = config.getfloat('TRAINING', 'Initial Learning Rate')
    trainer = ModelTrainer(
            model,
            *data_loaders,
            dataset_generator.ed_labels,
            dataset_generator.doc_indices,
            dataset_generator.doc_pos,
            learning_rate,
            epochs,
            args.gpu_index
        )
    print(f"Initial learning rate: {trainer.scheduler.get_last_lr()[0]:.4e}")

    print("\n 5. Loading knowledgebase for evaluation")
    kb = get_knowledge_base(config, wiki='pedia', with_cg=True)

    # kb_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']
    # w2vec_file = path.join(kb_dir, config['KNOWLEDGE BASE']['Wikipedia2Vec File'])

    # alias_dict_file = ''
    # entity_dict_file = ''
    # p_e_m_file = ''
    # if config.has_option('KNOWLEDGE BASE', 'Alias Dict') \
    #         and config.has_option('KNOWLEDGE BASE', 'Entity Dict'):
    #     alias_dict_file = config['KNOWLEDGE BASE']['Alias Dict']
    #     entity_dict_file = config['KNOWLEDGE BASE']['Entity Dict']
    # else:
    #     p_e_m_file = config['KNOWLEDGE BASE']['Alias Mapping']

    # kb = KnowledgeBaseWikipedia(
    #         w2vec_file,
    #         alias_dict_file=alias_dict_file,
    #         entity_dict_file=entity_dict_file,
    #         p_e_m_file=p_e_m_file,
    #     )

    # # print("\n 5a. Loading Alias Mapping for "
    # #       "candidate generation during evaluation")
    # # kb.init_alias_mapping(config['KNOWLEDGE BASE']['Alias Mapping'])

    print("\n 6. Starting training")
    early_stopping = config.getboolean('TRAINING', 'Early Stopping')
    train_update_freq = config.getint('VERBOSITY', 'Training Update Frequency')
    val_update_freq = config.getint('VERBOSITY', 'Validation Update Frequency')
    save_dir = config['MODEL']['Save Model Dir']
    train_start = time.strftime('%Y%m%d_%H%M', time.gmtime(time.time()))
    save_dir = path.join(save_dir, train_start)
    save_freq = args.checkpoint
    training_stats = trainer.train(
            kb,
            early_stopping,
            train_update_freq,
            val_update_freq,
            save_freq,
            save_dir,
            docs_file,
        )

    print(f"\n 7. Saving final model to directory '{save_dir}'")
    model_dir = save_model_to_dir(model, save_dir)
    write_stats_to_file(training_stats, model_dir)

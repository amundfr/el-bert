"""
Script to train a BERT model.
Assumes data has been generated with generate_input_data.py
"""
import time
import argparse
import os
from configparser import ConfigParser
from src.toolbox import get_knowledge_base
from src.dataset_generator import DatasetGenerator
from src.bert_model import load_bert_from_file
from src.trainer import ModelTrainer
import transformers
transformers.logging.set_verbosity_error()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__
        )

    parser.add_argument(
            "model_dir", type=str, nargs='?',
            help="path to a folder with a trained model"
        )
    # parser.add_argument(
    #         "-s", "--subdirectories", action='store_true', default=False,
    #         help="flag: if provided, scan subdirectories of model_dir for "
    #              "models and evaluate them in turn"
    #     )
    parser.add_argument(
            "-n", "--evaluate_n_documents", type=int, nargs='?', default=0,
            help="the number of documents to evaluate from each dataset "
                 "(0 or nothing for all)"
        )
    parser.add_argument(
            "-d", "--dataset", type=str, nargs='+',
            choices=['test', 'val', 'train'],
            help="datasets to evaluate on. "
                 "Uses dataset split instructions from Config.ini"
        )
    parser.add_argument(
                "-g", "--gpu_index", type=int, nargs='?', default=0,
                help="the index of the GPU to use"
            )
    parser.add_argument(
                "-c", "--candidate_sets", action='store_true', default=False,
                help="flag: if provided, use candidate sets for evaluation.",
            )
    parser.add_argument(
                "-f", "--print_to_file", action='store_true', default=False,
                help="flag: if provided, results are printed to file "
                     "in model directory"
            )
    parser.add_argument(
                "--print_docs", action='store_true', default=False,
                help="flag: if provided, results on all documents are printed"
            )
    parser.add_argument(
                "--no_eval_unseen", action='store_false', default=True,
                help="flag: if provided, do not evaluate "
                     "performance on entities that"
                     " appear in the training set, and entities that do not"
            )
    parser.add_argument(
                "-k", "--skip_evaluated", action='store_true', default=False,
                help="flag: if provided skip evaluation of models on datasets "
                     "if they have been evaluated on that dataset before"
            )

    args = parser.parse_args()

    print("\nStarting evaluation script."
          "\nThis script relies on paths in 'config.ini',"
          "\n and requires the data from generate_input_data.py")

    print("\n 1. Getting 'config.ini'")
    config = ConfigParser()
    config.read('config.ini')

    print("\n 2. Loading knowledgebase")
    if 'wikipedia' in config['DATA']['Annotated Dataset']:
        kb_is_wikipedia = True
    else:
        kb_is_wikipedia = False

    if args.candidate_sets:
        # Use full Wikipedia2vec as KB with Candidate Generation
        kb_dir = config['KNOWLEDGE BASE']['Wikipedia2vec Directory']
    elif kb_is_wikipedia:
        # Use compact Wikipedia2vec as KB without Candidate Generation
        kb_dir = config['KNOWLEDGE BASE']['Compact Wikipedia KB']
    else:
        kb_dir = config['KNOWLEDGE BASE']['Compact Wikidata KB']

    # Where to save evaluation results
    if args.print_to_file:
        results_dir = os.path.split(kb_dir)
        if results_dir[-1] == '':
            results_dir = os.path.split(results_dir[0])
        results_dir = results_dir[-1]

    t0 = time.time()
    if kb_is_wikipedia:
        print(
                f"Loading Wikipedia KB with"
                f"{'out' if not args.candidate_sets else ''}"
                f" Candidate Generation"
            )
        kb = get_knowledge_base(
                config,
                wiki='pedia',
                with_cg=args.candidate_sets
            )
    else:
        print(
                f"Loading Wikidata KB with"
                f"{'out' if not args.candidate_sets else ''}"
                f" Candidate Generation"
            )
        kb = get_knowledge_base(
                config,
                wiki='data',
                with_cg=args.candidate_sets
            )

    print(f"Initializing KB took {time.time() - t0:.1f} seconds")

    model_dir = args.model_dir

    if model_dir is None:
        model_dir = input("\nEnter model path, or empty to quit evaluation: ")
    # while model_dir != '':
    #     # If there are subdirectories, and instructions to use them
    #     if args.subdirectories and any(
    #                 [os.path.isdir(os.path.join(model_dir, p))
    #                  for p in os.listdir(model_dir)]
    #             ):
    #         print(f"Using subdirectories of {model_dir}")
    #         model_dirs = [
    #                 os.path.join(model_dir, p)
    #                 for p in os.listdir(model_dir)
    #                 if os.path.isdir(os.path.join(model_dir, p))
    #             ]
    #     else:
    #         model_dirs = [model_dir]
    #     print(f"Model queue: {model_dirs}")
    #     for model_dir in model_dirs:
    print(f"\n 3. Loading BERT from directory '{model_dir}'")
    try:
        model = load_bert_from_file(model_dir)
    except FileNotFoundError as ex:
        print(ex)
    print(f"{'U' if model.config.use_dropout else 'Not u'}"
          f"sing dropout after BERT Embeddings")
    print(f"{'U' if model.config.hidden_output_layers else 'Not u'}"
          f"sing hidden output layers")

    if model.config.vocab_size == 28996:
        vectors_dir = config['DATA']['Cased Input Vectors Dir']
    else:
        vectors_dir = config['DATA']['Uncased Input Vectors Dir']
    print(f"\n 4. Loading dataset from '{vectors_dir}'"
          f", and splitting data")
    dataset_generator = DatasetGenerator.load(vectors_dir)
    batch_size = config.getint('TRAINING', 'Batch size')
    split = config['DATA']['Data Split'].split(', ')
    split = [int(s) for s in split]
    print(f"Using data split: {split}")
    data_loaders = dataset_generator.split_by_ratio(split, batch_size)
    documents_file = config['DATA']['Annotated Dataset']

    print("\n 5. Getting model trainer")
    trainer = ModelTrainer(
            model,
            *data_loaders,
            ed_labels=dataset_generator.ed_labels,
            dataset_to_doc=dataset_generator.doc_indices,
            dataset_to_doc_pos=dataset_generator.doc_pos,
            learning_rate=1,
            epochs=0,
            gpu_index=args.gpu_index
        )

    print("\n 6. Starting evaluation...")
    if args.dataset is None:
        args.dataset = ['test']
    for dataset in args.dataset:
        if args.print_to_file:
            res_out_dir = os.path.join(model_dir, results_dir)
        if args.skip_evaluated and args.print_to_file:
            # Skip if this evaluation has run before
            if os.path.isdir(res_out_dir):
                res_f = os.path.join(
                        model_dir, results_dir,
                        dataset + '_evaluation_'
                    )
                res_f += 'seen_unseen_' if args.no_eval_unseen else ''
                res_f += \
                    'cs.txt' if args.candidate_sets else 'no_cs.txt'
                if os.path.isfile(res_f):
                    print(f"Found file {res_f} from "
                          "previous evaluation. Skipping ...")
                    continue
        print(f"\n ... Evaluating {dataset} dataset")
        t0 = time.time()
        print(f"Starting at {time.ctime(t0)}")
        print(f"\nResults on {dataset}")
        _, evaluator, _, _ = trainer.evaluate(
                dataset=dataset,
                knowledgebase=kb,
                sample_n=args.evaluate_n_documents,
                update_freq=config.getint(
                        'VERBOSITY', 'Test Update Frequency'
                    ),
                verbose=False,
                candidate_generation=args.candidate_sets,
                documents_file=documents_file
            )
        t1 = time.time()
        if args.print_docs:
            evaluator.print_documents()
        eval_str = evaluator.evaluation_str()
        print()
        print(eval_str)
        d_t_str = \
            time.strftime("%H:%M:%S hh:mm:ss", time.gmtime(t1-t0))
        print(f"  Took: {d_t_str}\n----")

        if args.print_to_file:
            dump_file_name = "2_conll_" + dataset
            if not args.candidate_sets:
                dump_file_name += '_no'
            dump_file_name += '_cs'
            evaluator.dump_pred_to_file(
                    model_dir + '/predictions',
                    dump_file_name,
                )

        # Write evaluation result, prediction and labels to files
        if args.print_to_file:
            res_file_eval = os.path.join(
                    res_out_dir, dataset + '_evaluation'
                )
            if args.candidate_sets:
                res_file_eval += '_cs'
            else:
                res_file_eval += '_no_cs'
            res_file_eval += '.txt'
            print(
                eval_str,
                file=open(res_file_eval, 'w')
            )
        if args.no_eval_unseen and not dataset == 'train':
            train_end = len(trainer.train_dataloader.dataset.indices)
            ed_labels = trainer.ed_labels[:train_end]
            eval_unseen_str, _ = evaluator.eval_seen_unseen(ed_labels)
            print()
            print(eval_unseen_str)
            if args.print_to_file:
                res_file_unseen = os.path.join(
                        res_out_dir,
                        dataset + '_evaluation_seen_unseen'
                    )
                if args.candidate_sets:
                    res_file_unseen += '_cs'
                else:
                    res_file_unseen += '_no_cs'
                res_file_unseen += '.txt'
                print(
                    eval_unseen_str,
                    file=open(res_file_unseen, 'w')
                )

        # model_dir = ""
        # model_dir = input("\nEnter model path, or empty to quit: ")
        # model_dir = model_dir.strip()

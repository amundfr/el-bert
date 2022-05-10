# el-bert = Entity Linking BERT
Entity Linking with BERT through joint modeling of mention detection and entity disambiguation.

## Requirements

This project has two companion Makefiles: one to build and run the Docker container, and one to run the Python code accessed from inside the container.

You will need a machine with CUDA to run this code. The Docker container is configured to use CUDA 10.1, cuDNN 7 (following the AD machine "Flavus").

The Docker container also relies on files and folders located at /nfs/students/amund-faller-raheim/ma_thesis_el_bert

## Changing Settings of the Project

Most scripts read from the 'config.ini' file located at the root of the project. To change the behavious of the program, it is recommended to look here before the CLI arguments of each script. Particularly with regard to training behaviour. 

For example, to pretrain a model on a different dataset, change the path at "DATA" -> "Annotated Dataset" and "DATA" -> "[Cased/Uncased] Input Vectors Dir" and make sure you are using a Knowledge Base with the same IDs as the dataset (i.e. Wikidata or Wikipedia). Update the "DATA" -> "Data Split" to split the dataset as prefered.

### Build and run the Docker Container

From the root of the project folder (where this file is located), run

```bash
make build && make run
```

These commands will run _Wharfer_ with correct paths and container names, and start the container. 

### Running the Main Scripts

#### 1. Generate Missing Data 

Inside the container, you will be greeted by a new Makefile. To generate missing files, run
```bash
make setup
```
This will run three scripts to generate missing files:
 * the Candidate Generation files /ex_data/entity_dict.json and /ex_data/alias_dict.json. Generated from the file at config's "KNOWLEDGE BASE" -> "Alias Mapping", which should point to the file prob_yago_crosswikis_wikipedia_p_e_m.txt with candidate sets from Ganea & Hofmann (2017);
 * Wikipedia annotated AIDA-CoNLL file /ex_data/annotated_datasets/conll-wikipedia-iob-annotations. Generated from a Wikidata annotated AIDA-CoNLL file, a file /ex_data/AIDA-YAGO2-annotations.tsv with Wikidata and Wikipedia annotations, and an additional mapping from Wikidata to Wikipedia located at /ex_data/wikidata_wikipedia_mapping.csv;
 * a compact Wikipedia2vec Knowledge Base at /ex_data/knowledgebases/wikipedia_score_15 used when evaluating without Candidate Generation. Generated from a Wikipedia2vec file defined by config's "KNOWLEDGE BASE" -> "Wikipedia2vec Directory".

#### 2. Generate Training Data Vectors
To generate the data vectors digested by the model, run
```bash
make data-generation
```

This script uses the file defined by config's "DATA" -> "Annotated Dataset" to generate a vectorized data as digested by BERT. The script will use the tokenizer for the model defined by "MODEL" -> "Model ID" and infer "cased" or "uncased". The script writes to "DATA" -> "(Un)Cased Input Vectors Dir".

#### 3. Run unittests

To run unittest
```bash
make unittest
```

#### 4. Training a New Model
To train a new model, run
```bash
make train
```

The model architecture is defined by parameters at config's "MODEL" -> "Model ID", "Hidden Output Layers", "Dropout After BERT".

The training procedure is influenced by the parameters at config's "TRAINING" -> "Epochs", "Batch Size", "Initial Learning Rate", "Loss Lambda", "Early Stopping".

The new model will be saved in /models/trained/\<train-start\>/\<checkpoint-saved\>


#### 5. Evaluating the Latest Model

To evaluate the lastest trained model on AIDA-CoNLL Test with Candidate Generation, run
```bash
make eval-test
```

Or on the AIDA-CoNLL Validation dataset
```bash
make eval-val
```

Or without Candidate Generation (takes much longer)
```bash
make eval-test-nocg
```
```bash
make eval-val-nocg
```

### Evaluate Main Models

To evaluate the most important models in the thesis, run the following commands (assuming you have access to the models). 

Removing the flag '-c' will evaluate without Candidate Generation.

 * The base model (best performing without pretraining):
```bash
python3 evaluate_model.py -c -d test --no_eval_unseen "/models/trained/5_4_3-table_11-base_model/epoch_180"
```
 * The pretrained model:
```bash
python3 evaluate_model.py -c -d test --no_eval_unseen "/models/trained/5_4_3-table_11-pretrained_model/epoch_180"
```
 * Our version of the model of Chen et al.:
```bash
python3 evaluate_model.py -c -d test --no_eval_unseen "/models/trained/6_1_1-table_12-our_chen/epoch_180"
```

### Additional Evaluation

There are a number of evaluation scripts used to harvest statistics for the thesis. These can be accessed with `make` as well.

For Section 5.2.2 in thesis:
```bash
make compare-datasets
```

For Section 5.3.1 in thesis:
```bash
make evaluate-cg
```

For Section 5.3.2 in thesis:
```bash
make evaluate-kb
```

For Section 6.2.1 in thesis:
```bash
make evaluate-by-cat
```

For Section 6.2.1 in thesis:
```bash
make evaluate-unseen
```

For Section 6.2.2 in thesis:
```bash
make popularity-corr
```

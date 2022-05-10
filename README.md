# el-bert = Entity Linking BERT
Entity Linking with BERT through joint modeling of mention detection and entity disambiguation.

## Requirements

This project has two companion Makefiles: one to build and run the Docker container, and one to run the Python code accessed from inside the container.

You will need a machine with CUDA to run this code. The Docker container is configured to use CUDA 10.1, cuDNN 7 (following the AD machine "Flavus").

The Docker container also relies on files and folders located at /nfs/students/amund-faller-raheim/ma_thesis_el_bert

### Build and run the Docker Container

From the root of the project folder (where this file is located), run

```bash
make build && make run
```

### Running the Main Scripts

Inside the container, you will be greeted by a new Makefile. To generate missing files, run
```bash
make setup
```

To generate the data vectors digested by the model, run
```bash
make data-generation
```

To run unittest
```bash
make unittest
```

To train a new model, run
```bash
make train
```

To evaluate the lastest trained model, run
```bash
make evaluate-test
```

Alternatively, the following command is a shorthand for all of the above
```bash
make full
```

### Additional Evaluation

There are a number of evaluation scripts used to harvest statistics for the thesis. These can be accessed with `make` aswell.

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

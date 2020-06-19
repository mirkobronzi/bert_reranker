# Embedding-based Bert Re-Ranker

## Info
This repository contains the code to train/use a deep learning model which
solves the following task: given a question `q` , and N candidate master
questions `Q`, find the master question in `Q` that is semantically closer
to `q`.

In short, this is done by using a model to embed the various questions
(usually BERT), so that every question is represented by a vector.
Then a dot-product can be used to compute the similarity of the vectors/questions.

### Out-of-domain questions (or outliers)

This repository also provide the code to spot out-of-domain (or outlier) questions.
That is, if the the question `q` has no similar master question in `Q`, then
the model will predict that `q` is an outlier.

This is done using an outlier detector model.
So, putting everything together, the process is basically a pipeline:
 - run the outlier detector model. If `q` is an outlier, return this, otherwise
 - run the main model and find the closest element in `Q`, and return it.
 
## Data

The expected data format is a json file that is a dictionary with two components:
`examples` and `passages`.
You can see an example in `examples/local/data_small.json`.

### Examples

An example is a user question that we want to match. The main important fields are:
- `question`: which represent the user question.
- `passage_id`: which is the id of the passage that is the best match for this example.

### Passages

A passage represent a master question (i.e., an element from the `Q` set - see above).
It contains several information; the most important ones are:
 - the `passage_id` (used as identifier)
 - the `reference_type` - used to mark the passages as ID (in this case the `reference_type` 
 starts with “faq”) or OOD (in this case the `reference_type` does not start with “faq”)
 - the `section_headers`: the last element in this list represents the text of the master question.

### Sources

Both examples and passages have a `source` field. This is used to match an example to the possible passages.
For example, if a user is direct to the FAQ Quebec website, only the passages that are from the 
FAQ Quebec website are considered as candidates.

Thanks to this `source` field, we can handle multiple sources in our model.

(more in details, have a single joint model that can serve more sources, instead of training 
a separate model per source)

## How to run

From the root of the repository:

    pip install -e .

This will install all the dependencies.
To run the code, there is an example in the following folder:

    cd examples/local
    sh run.sh

This will run the train phase.

The code mainly supports 3 operations:
 - train: will train a model from scratch.
 - predict: will use an already trained model to compute predictions on some data.
 - generate embeddings: will run on some data and generate the relative embeddings.

Note that all the 3 operations are mostly based on a config file that specify the
model's architecture. You can see an example in `examples/local/config.yaml`.

### Train

To train, you can use the following command:

    main --config config.yaml --output output --train

This will create a new model in the `output` folder.

Use `--help` to see the other available options.

### Predict

After training, you can use a model to predict (and evaluate) on a json file with the format
that was described above. To do this:

    main --config config.yaml --output output --predict data_small.json --predict-to predictions.txt

This will use the model store in `output` to generate predictions on the file `data_small.json`, and
the results will be store in the file `predictions.txt`.

Note the `config.yaml` file is the same as the one used in training.

Use `--help` to see the other available options.

### Generate embeddings

This step can be used (after training) to generate embeddings for some `examples` and/or `passages`.
This can be useful for training the outlier detector. (see next section)

To do this, the command is:

    main --config config.yaml --output output --file-to-emb=data_small.json --write-emb-to=emb.npy

This will use the model in `output` to generate embeddings for the data in `data_small.json`, and these
embeddings will be written to `emb.npy`.

Use `--help` to see the other available options.

### Train the outlier detector

To train the outlier detector (which is a sklearn model), you will need to generate the embedding
first (see previous section).

One this is done, you can train the outliser model as:

    train_outlier_detector --embeddings=emb.npy --output=output --train-on-passage-headers

This will use the embeddings in `emb.npy` to train the model. In this example, only the passages
will be used (see `--train-on-passage-headers`). To use also the user questions to train the model,
the command is:

    train_outlier_detector --embeddings=emb.npy --output=output --train-on-passage-headers --train-on-questions

Use `--help` to see the other available options.

### Hyper-parameter search with Orion

See the example under if you want to run an hyper-parameter search procedure:

    cd examples/local_orion
    sh run.sh

Note that this will be done by using Orion (https://github.com/Epistimio/orion).

## Code structure

The file `bert_reranker/main.py` is the main entry point to run the operations above.
It mainly takes care of assembling together the various part of the code, in particular:
 - the code to load the data
 - the code to create the models
 - the code to train the models

### Data loading

Data is handled in `bert_reranker/data/data_loader.py`. This file provides a PyTorch DataSet
implementation that is able to wrap the data in the format we specified above.

This file also contains all the utilities to deal with the this data format.
It would be best to re-use them instead to deal with the json directly (so that we can keep
all the procedure in a centralized place).

### Creating the models

Models are created by using the file `bert_reranker/models/load_model.py`.
This file is mainly a dispatcher that will call the appropriate method to for the model that
the user want to use. In particular, most of the time the model that will be used will be BERT,
and the related code is in `bert_reranker/models/bert_encoder.py`. Note that this file contains
various version of the BERT encoder (e.g., a vanilla version, a version that supports caching the 
results, ...).

An encoder can be used on the examples to get the related embeddings, as well as on the passages (
to get the embeddings). Note that the same encoder can be used for both (this is configurable in
the config file).

Once two encoders are created (or one only if the user decided to use the same one for example/passage),
then a retriever model (`bert_reranker/models/retriever.py`) is create by composing the two.
A retriever is indeed able to use the two encoders to generate the various embeddings and it will
then produce a score by performing a simple dot-product.

### Training the models

The file `bert_reranker/models/retriever_trainer.py` takes care of training the models.
We use PyTorch Lightning to help with training. Because of that, the retriever_trainer will subclass
`pl.LightningModule` (from PyTorch Lightning), and only implement the `train_step` and
`validate_step` methods.

In general, you can refer to the PyTorch Lightning for more info:
https://github.com/PyTorchLightning/pytorch-lightning .

### To contribute:
Enable flake8 check before commit:
(run from the root of the repository)

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

### To release:
(see https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56 )
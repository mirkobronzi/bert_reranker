# Embedding-based Bert Re-Ranker

## What is it
This repos contains the code to train/use a deep learning model which
solves the following task: given a question `q` , and N candidate master
questions `Q`, find the master question in `Q` that is semantically closer
to `q`.

In short, this is done by using a model to embed the various questions
(usually BERT), so that every question is represented by a vector.
The a dot-product can be used to compute the similarity of the vectors/questions.

### Out-of-domain

This repos also provide the code to spot out-of-domain (or outlier) questions.
That is, if the the question `q` has no similar master question in `Q`, then
the model will predict that `q` is an outlier.

This is done using an outlier detector model. The process is basically a pipeline:
 - run the outlier detector model. If `q` is an outlier, return this, otherwise
 - run the main model and find the closest element in `Q`, and return it.

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

### Hyper-parameter search with Orion

See the example under if you want to run an hyper-parameter search procedure:

    cd examples/local_orion
    sh run.sh

Note that this will be done by using Orion (https://github.com/Epistimio/orion).

### To contribute:
Enable flake8 check before commit:
(run from the root of the repository)

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

~~~~### To release:
(see https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56 )
# Embedding-based Bert Re-Ranker


Bert model to rank answers for a given question.

## Instructions

From the root of the repository:

    pip install -e .

Then:

    cd examples/local
    sh run.sh

## Hyper-parameter search with Orion

See the example under:

    cd examples/local_orion
    sh run.sh

Note the setup in `run.sh` will store the Orion files locally.

To use an online DB (e.g., to run on multiple machines), please follow
https://orion.readthedocs.io/en/latest/install/database.html#atlas-mongodb

### To contribute:
Enable flake8 check before commit:
(run from the root of the repository)

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -

### To release:
(see https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56 )

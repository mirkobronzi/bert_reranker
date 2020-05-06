# Embedding-based Bert Re-Ranker


Bert model to rank answers for a given question.

## Instructions

From the root of the repository:

    pip install -e .

Then:

    cd examples/local
    sh run.sh

### To contribute:
Enable flake8 check before commit:
(run from the root of the repository)

    cd .git/hooks/ && ln -s ../../config/hooks/pre-commit . && cd -
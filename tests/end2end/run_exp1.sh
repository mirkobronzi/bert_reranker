# exit at the first error
set -e
# go to the test folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/../../examples/local/

main --config config.yaml --output output --train
# TODO: fix the following one
#main --config config.yaml --output output --validate
main --config config.yaml --output output --predict data_small.json
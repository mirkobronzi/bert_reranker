# exit at the first error
set -e
# go to the test folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/../../examples/local

main --config config.yaml --output output --train
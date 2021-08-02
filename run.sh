#!/bin/bash
# Usage:
# ./run.sh case [debug]
#
# Eg:
# $ ./run.sh 1       # Run baseline
# $ ./run.sh 2 debug # Debug mode

CLEAR='\033[0m'
RED='\033[0;31m'

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED} $1${CLEAR}\n";
  fi
  exit 1
}

# Debug or not
BASE='python main.py'
if [ -n "$2" ]; then
    if [[ $2 == *"debug"* ]]; then
        BASE='python -m pudb main.py'
    fi
fi

case "$1" in
        1)
            echo "Small model to test"
            eval $BASE \
                --embed_dim 36 \
                --dec_input_size 5 \
                --dec_h_size 5 \
                --emb_chunks 3 \
                --num_embeddings 10 \
                --print_every 10 \
                --num_hiddens 8 \
                --batch_size 128 \
                --num_epochs 200 \
                --toy_dataset_size 10000

            ;;

        2)
            echo "Very small model to test"
            eval $BASE \
                --num_hiddens 10 \
                --embed_dim 36 \
                --emb_chunks 3 \
                --num_embeddings 10 \
                --dec_input_size 5 \
                --dec_h_size 5 \
                --batch_size 1 \
                --num_epochs 100 \
                --toy_dataset_size 10000

            ;;

        3)
            echo "Non-mon"
            eval $BASE \
                --nlayers 1 \
                --epoch 1000
            ;;

        *)
            usage "You need to call $0 with an int option"
            exit 1
esac


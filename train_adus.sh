#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

ELLOGON_TOOL_API="$SCRIPT_DIR/ellogon-tool"
SCRIPTS_DIR="${SCRIPT_DIR}/scripts"

# Retrieve data from the annotation tool...
DATA_ADUS="data/adu"
mkdir -p $DATA_ADUS

COLLECTION_NAMES=("Manual-22-docs-covid-crete-politics" "DebateLab 1 Kasteli")
#COLLECTION_NAMES=("docs22")
generate_data=false
#generate_data=true

if $generate_data ; then
  echo "Generating data from Collections: ${COLLECTION_NAMES[@]}"
  python $ELLOGON_TOOL_API/col2adu_conll.py "${COLLECTION_NAMES[@]}" > \
         $DATA_ADUS/data.csv
fi

# Train...
export CUDA_VISIBLE_DEVICES=1
rm -rf test-ner test-adu
python "$SCRIPTS_DIR/train_ADUs.py"

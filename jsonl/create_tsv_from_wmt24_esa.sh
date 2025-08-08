#!/bin/bash

set -e

# set HF_HOME to login
if [ -z $HF_HOME ]; then
    echo "Please set \`HF_HOME\` variable using the command \`export HF_HOME=<cache_dir>\`"
    exit 1
fi

echo "HF_HOME=$HF_HOME"
if [ ! -d $HF_HOME ]; then
    echo "$HF_HOME is not a directory"
    exit 1
fi

if ! command -v pipx &> /dev/null; then
    pip install pipx --user
    pipx ensurepath
fi

if ! command -v uv &> /dev/null; then
    pipx install uv
fi

# Create a tsv format file for wmt24_esa file from wmt24_esa.jsonl
WMT24_ESA_JSONL=wmt24_esa.jsonl
REFERENCES_TSV=wmt24_esa.tsv

uv run --python 3.11 \
    --with pandas \
    --with datasets \
    create_tsv_from_wmt24_esa.py \
    --wmt24_esa_jsonl $WMT24_ESA_JSONL \
    --output_tsv $REFERENCES_TSV \
    --filter_data_with_invalid_span

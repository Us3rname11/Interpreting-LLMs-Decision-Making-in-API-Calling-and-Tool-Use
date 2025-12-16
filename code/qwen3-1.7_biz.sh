#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

pip install -r pegasus-requirements.txt

pip uninstall -y flash-attn
pip install flash-attn --no-build-isolation

python3 main.py \
    --category business_and_productivity \
    --model_name "Qwen3-1.7B" \
    --attribution "integrated_gradients"

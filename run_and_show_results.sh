#!/bin/bash
# LiveBench Runner: ./run_and_show_results.sh <model> <benchmark> [parallel]

set -e

[ $# -lt 2 ] && { echo "Usage: $0 <model> <benchmark> [parallel]"; exit 1; }

MODEL=$1
BENCHMARK=$2
PARALLEL=${3:-1}
RELEASE="2024-11-25"

# Checks
[ -z "$VIRTUAL_ENV" ] && source .venv/bin/activate
curl -s http://localhost:11434/api/tags >/dev/null || { echo "Start Ollama first"; exit 1; }

# Run
cd livebench
python run_livebench.py \
    --model "$MODEL" \
    --bench-name "$BENCHMARK" \
    --parallel-requests "$PARALLEL" \
    --livebench-release-option "$RELEASE" \
    --api-base "http://localhost:11434/v1" \
    --question-source "jsonl"

# Results
[ -d "data/$BENCHMARK/model_judgment" ] || { echo "No results"; exit 1; }

python show_livebench_result.py \
    --bench-name "$BENCHMARK" \
    --model-list "$MODEL" \
    --question-source "jsonl" \
    --livebench-release-option "$RELEASE"

# Local LiveBench Evaluation with Ollama

Quick guide to evaluate LLMs locally using Ollama and simplified bash scripts.

## Prerequisites

- [Ollama](https://ollama.ai) installed
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer
- Python 3.8+
- Git

## Quick Start

### 1. Setup Environment

```bash

# Clone and enter directory
git clone https://github.com/BaklazhenkoNikita/local_eval.git
cd local_eval

# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Start Ollama

```bash
# Start Ollama server
./start_ollama.sh serve

# Check status
./start_ollama.sh status

# Pull a model (if needed)
./start_ollama.sh pull qwen2.5:3b
./start_ollama.sh pull llama3.2:1b

#Stop Ollama
./kill_benchmarks.sh
brew services stop ollama
```

### 3. Download Questions

```bash
cd livebench
python download_questions.py --livebench-release-option 2024-11-25
cd ..
```

### 4. Run Benchmarks

```bash
# Basic usage: ./run_and_show_results.sh <model> <benchmark> [parallel]
./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/tablejoin 5
```

## Benchmark Examples

### Reasoning Tasks

```bash
# Web of Lies
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/web_of_lies_v2 5

# Zebra Puzzle
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/zebra_puzzle 5

# Spatial Reasoning
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/spatial 5
```

### Math Tasks

```bash
# AMPS Hard
./run_and_show_results.sh qwen2.5:3b live_bench/math/AMPS_Hard 5

# Math Competitions
./run_and_show_results.sh qwen2.5:3b live_bench/math/math_comp 5

# Olympiad
./run_and_show_results.sh qwen2.5:3b live_bench/math/olympiad 5
```

### Custom Tests

```bash
# Basic Math (custom test - addition, subtraction, multiplication, division, fractions, percentages, powers)
./run_and_show_results.sh qwen2.5:3b custom_tests/basic_math 4

# You can create your own custom tests in data/custom_tests/
# See data/custom_tests/basic_math/ for example structure
```

### Coding Tasks

```bash
# Coding Completion
./run_and_show_results.sh qwen2.5:3b live_bench/coding/coding_completion 3

# LCB Generation
./run_and_show_results.sh qwen2.5:3b live_bench/coding/LCB_generation 3
```

### Data Analysis Tasks

```bash
# CTA (Column Type Annotation)
./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/cta 5

# Table Join
./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/tablejoin 5

# Table Reformat
./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/tablereformat 5
```

### Instruction Following

```bash
# Paraphrase
./run_and_show_results.sh qwen2.5:3b live_bench/instruction_following/paraphrase 5

# Simplify
./run_and_show_results.sh qwen2.5:3b live_bench/instruction_following/simplify 5

# Story Generation
./run_and_show_results.sh qwen2.5:3b live_bench/instruction_following/story_generation 5

# Summarize
./run_and_show_results.sh qwen2.5:3b live_bench/instruction_following/summarize 5
```

### Language Tasks

```bash
# Typos
./run_and_show_results.sh qwen2.5:3b live_bench/language/typos 5

# Connections
./run_and_show_results.sh qwen2.5:3b live_bench/language/connections 5

# Plot Unscrambling
./run_and_show_results.sh qwen2.5:3b live_bench/language/plot_unscrambling 5
```

## Running All Tasks

```bash
# Run all reasoning tasks
for task in web_of_lies_v2 zebra_puzzle spatial; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/$task 5
done

# Run all math tasks
for task in AMPS_Hard math_comp olympiad; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/math/$task 5
done

# Run custom tests
./run_and_show_results.sh qwen2.5:3b custom_tests/basic_math 4

# Run all coding tasks
for task in coding_completion LCB_generation; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/coding/$task 3
done

# Run all data analysis tasks
for task in cta tablejoin tablereformat; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/$task 5
done

# Run all instruction following tasks
for task in paraphrase simplify story_generation summarize; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/instruction_following/$task 5
done

# Run all language tasks
for task in typos connections plot_unscrambling; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/language/$task 5
done
```

## Testing Different Models

```bash
# Small models (faster)
./start_ollama.sh pull llama3.2:1b
./run_and_show_results.sh llama3.2:1b live_bench/reasoning/web_of_lies_v2 5

./start_ollama.sh pull qwen2.5:3b
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/web_of_lies_v2 5

# Medium models
./start_ollama.sh pull llama3.2:3b
./run_and_show_results.sh llama3.2:3b live_bench/math/AMPS_Hard 3

./start_ollama.sh pull qwen2.5:7b
./run_and_show_results.sh qwen2.5:7b live_bench/math/AMPS_Hard 3

# Large models (slower, more accurate)
./start_ollama.sh pull qwen2.5:14b
./run_and_show_results.sh qwen2.5:14b live_bench/coding/coding_completion 1

./start_ollama.sh pull llama3.1:8b
./run_and_show_results.sh llama3.1:8b live_bench/coding/LCB_generation 1
```

## Performance Tips

### Parallel Requests

Adjust the parallel parameter based on your hardware:

```bash
# Low-end hardware (4GB RAM)
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/web_of_lies_v2 1

# Mid-range hardware (8-16GB RAM)
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/web_of_lies_v2 3

# High-end hardware (32GB+ RAM)
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/web_of_lies_v2 10
```

### Model Selection by Task

**Reasoning & Math:** Larger models (7B+) perform significantly better
```bash
./run_and_show_results.sh qwen2.5:7b live_bench/math/olympiad 3
```

**Data Analysis:** Medium models (3B-7B) are sufficient
```bash
./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/tablereformat 5
```

**Coding:** Larger models (7B+) or code-specialized models recommended
```bash
./run_and_show_results.sh codellama:7b live_bench/coding/coding_completion 2
```

## Ollama Management Commands

```bash
# Start Ollama
./start_ollama.sh serve

# Check status and list models
./start_ollama.sh status

# List all downloaded models
./start_ollama.sh list

# Pull/download a model
./start_ollama.sh pull <model_name>

# Stop Ollama server
./start_ollama.sh stop
```

## Troubleshooting

### "Start Ollama first" error
```bash
./start_ollama.sh serve
```

### "No results" error
Download questions first:
```bash
cd livebench
python download_questions.py --livebench-release-option 2024-11-25
cd ..
```

### Model not found
Pull the model:
```bash
./start_ollama.sh pull qwen2.5:3b
```

### Out of memory errors
- Reduce parallel requests (use 1 or 2)
- Use smaller models
- Close other applications

### Slow evaluation
- Increase parallel requests (if you have RAM)
- Use smaller models for testing

## Results

Results are saved in:
```
livebench/data/<benchmark>/model_answer/<model>.jsonl
livebench/data/<benchmark>/model_judgment/<model>.jsonl
```

View results with:
```bash
cd livebench
python show_livebench_result.py \
    --bench-name live_bench/reasoning/web_of_lies_v2 \
    --model-list qwen2.5:3b \
    --question-source jsonl \
    --livebench-release-option 2024-11-25
```

## Available Models on Ollama

Popular models for evaluation:

| Model | Size | Best For | Command |
|-------|------|----------|---------|
| qwen2.5:0.5b | 0.5GB | Quick tests | `./start_ollama.sh pull qwen2.5:0.5b` |
| llama3.2:1b | 1.3GB | Fast evaluation | `./start_ollama.sh pull llama3.2:1b` |
| qwen2.5:3b | 1.9GB | Balanced | `./start_ollama.sh pull qwen2.5:3b` |
| llama3.2:3b | 2.0GB | General tasks | `./start_ollama.sh pull llama3.2:3b` |
| qwen2.5:7b | 4.7GB | Better accuracy | `./start_ollama.sh pull qwen2.5:7b` |
| llama3.1:8b | 4.7GB | High quality | `./start_ollama.sh pull llama3.1:8b` |
| qwen2.5:14b | 9.0GB | Best results | `./start_ollama.sh pull qwen2.5:14b` |
| codellama:7b | 3.8GB | Coding tasks | `./start_ollama.sh pull codellama:7b` |

See all models: https://ollama.ai/library

## How Benchmarks Are Evaluated

LiveBench uses deterministic evaluation methods for most categories:

### Reasoning
Extracts answers from `<solution>` tags or bold markdown (`**text**`), then compares to ground truth. Tasks include Web of Lies (yes/no/unknown matching), House Traversal (ordered name verification), Spatial (word-to-number conversion), and Zebra Puzzle (constraint satisfaction).

**Scoring:** Binary (0 or 1)

### Math
- **Competitions (AMC/AIME):** Parses `\boxed{}` LaTeX or multiple-choice letters
- **AMPS Hard:** Uses SymPy for symbolic equivalence checking, with LLM-as-judge fallback (OpenAI o3) when symbolic methods fail
- **Olympiad:** Edit distance for proof rearrangement

**Scoring:** Binary (0 or 1)

### Coding
Executes code against test cases with timeout/memory limits. Standard tasks use 20s timeout and 30MB limits. Agentic coding creates Docker environments and submits patches to real GitHub repos.

**Scoring:** Binary - passes all tests or not

### Data Analysis
- **Table Reformat:** Parses outputs as DataFrames (JSON/CSV/TSV/markdown), compares element-by-element with 1e-6 tolerance for numerics
- **CTA:** Normalizes and compares text strings
- **Table Join:** Calculates F1-score: `(2*TP) / (2*TP + FP + FN)`

**Scoring:** Binary for reformat/CTA; continuous (0-1) for table join

### Instruction Following
Checks each instruction using a registry of instruction classes. Computes two scores: (1) 1.0 if ALL followed, else 0.0, and (2) fraction of instructions followed. Final score is the average.

**Scoring:** Continuous (0-1)

### Writing
- **Typos:** Substring matching in `<solution>` tags
- **Plot Unscrambling:** Levenshtein distance on sentence orderings from `<PLOT_SUMMARY>` tags
- **Connections:** Puzzle-specific logic

**Scoring:** Binary for typos; continuous (0-1) for plot unscrambling

**Main evaluator:** `livebench/gen_ground_truth_judgment.py` routes questions to appropriate evaluators and uses ThreadPoolExecutor for parallel processing.

## Support

For issues with:
- **Original LiveBench:** https://github.com/LiveBench/LiveBench
- **This fork:** https://github.com/BaklazhenkoNikita/local_eval
- **Ollama:** https://github.com/ollama/ollama

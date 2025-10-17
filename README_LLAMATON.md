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
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter directory
git clone https://github.com/BaklazhenkoNikita/local_eval.git
cd local_eval

# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

> **Why uv?** `uv` is 10-100x faster than `pip` for package installation and resolution. Perfect for quick setup!

### 2. Start Ollama

```bash
# Start Ollama server
./start_ollama.sh serve

# Check status
./start_ollama.sh status

# Pull a model (if needed)
./start_ollama.sh pull qwen2.5:3b
./start_ollama.sh pull llama3.2:1b
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
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/web_of_lies_v2 50
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

# House Traversal
./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/house_traversal 5
```

### Math Tasks

```bash
# AMPS Hard
./run_and_show_results.sh qwen2.5:3b live_bench/math/AMPS_Hard 5

# Math Competitions
./run_and_show_results.sh qwen2.5:3b live_bench/math/math_competitions 5

# Olympiad
./run_and_show_results.sh qwen2.5:3b live_bench/math/olympiad 5
```

### Coding Tasks

```bash
# Code Completion
./run_and_show_results.sh qwen2.5:3b live_bench/coding/code_completion 3

# Code Generation
./run_and_show_results.sh qwen2.5:3b live_bench/coding/code_generation 3

# LCB Code Generation
./run_and_show_results.sh qwen2.5:3b live_bench/coding/lcb_code_generation 3
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
# Instruction Following
./run_and_show_results.sh qwen2.5:3b live_bench/instruction_following/instruction_following 5
```

### Writing Tasks

```bash
# Typos
./run_and_show_results.sh qwen2.5:3b live_bench/writing/typos 5

# Connections
./run_and_show_results.sh qwen2.5:3b live_bench/writing/connections 5

# Plot Unscrambling
./run_and_show_results.sh qwen2.5:3b live_bench/writing/plot_unscrambling 5
```

## Running All Tasks

```bash
# Run all reasoning tasks
for task in web_of_lies_v2 web_of_lies_v3 zebra_puzzle spatial house_traversal; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/reasoning/$task 5
done

# Run all math tasks
for task in AMPS_Hard math_competitions olympiad; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/math/$task 5
done

# Run all data analysis tasks
for task in cta tablejoin tablereformat; do
    ./run_and_show_results.sh qwen2.5:3b live_bench/data_analysis/$task 5
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
./run_and_show_results.sh qwen2.5:14b live_bench/coding/code_completion 1

./start_ollama.sh pull llama3.1:8b
./run_and_show_results.sh llama3.1:8b live_bench/coding/code_generation 1
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
./run_and_show_results.sh codellama:7b live_bench/coding/code_completion 2
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
- Use GPU if available (Ollama automatically uses GPU)

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

## Support

For issues with:
- **Original LiveBench:** https://github.com/LiveBench/LiveBench
- **This fork:** https://github.com/BaklazhenkoNikita/local_eval
- **Ollama:** https://github.com/ollama/ollama


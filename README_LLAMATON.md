# Local LiveBench Evaluation with Ollama

Quick guide to evaluate LLMs locally using Ollama with Python scripts and VS Code tasks.

## Prerequisites

- [Ollama](https://ollama.ai) installed
- [uv](https://docs.astral.sh/uv/) or pip
- Python 3.8+
- Git
- VS Code (optional, for task integration)

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
ollama serve

# Check status
ollama ps

# Pull a model (if needed)
ollama pull qwen2.5:3b
ollama pull llama3.2:1b

# Stop Ollama
./kill_benchmarks.sh
brew services stop ollama
```

### 3. Download Questions

```bash
cd livebench
python3 download_questions.py --livebench-release-option 2024-11-25
cd ..
```

### 4. Run Benchmarks

**Using Python Script:**
```bash
# Basic usage
python3 run_and_show_results.py --model qwen2.5:3b --benchmark live_bench/data_analysis/tablejoin

# With custom parallel requests (default: 4)
python3 run_and_show_results.py --model qwen2.5:3b --benchmark live_bench/data_analysis/tablejoin --parallel 8

# Short form
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/basic_math -p 4
```

**Using VS Code Tasks:**
```
1. Open project in VS Code
2. Press Cmd+Shift+B (Mac) or Ctrl+Shift+B (Windows/Linux)
3. Select "LiveBench: Run Benchmark Group" or "LiveBench: Run Specific Benchmark"
4. Choose your model and benchmark from the dropdown
```

Available tasks:
- **LiveBench: Run Benchmark Group** - Run main categories (custom_tests, live_bench/reasoning, etc.)
- **LiveBench: Run Specific Benchmark** - Run individual tests with full dropdown list

## Benchmark Examples

### Custom Tests

```bash
# Basic Math (binary scoring - addition, subtraction, multiplication, division, fractions, percentages, powers)
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/basic_math

# Advanced Math (algebra, calculus, geometry, etc.)
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/advanced_math

# Translation (continuous scoring - Spanish/French/German to English translation)
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/translation

# You can create your own custom tests in livebench/data/custom_tests/
# See livebench/data/custom_tests/basic_math/ for binary scoring example
# See livebench/data/custom_tests/translation/ for continuous scoring example
```

### Reasoning Tasks

```bash
# Web of Lies
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/reasoning/web_of_lies_v2

# Zebra Puzzle
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/reasoning/zebra_puzzle

# Spatial Reasoning
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/reasoning/spatial
```

### Math Tasks

```bash
# AMPS Hard
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/math/AMPS_Hard

# Math Competitions
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/math/math_comp

# Olympiad
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/math/olympiad
```

### Coding Tasks

```bash
# Coding Completion
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/coding/coding_completion

# LCB Generation
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/coding/LCB_generation
```

### Data Analysis Tasks

```bash
# CTA (Column Type Annotation)
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/data_analysis/cta

# Table Join
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/data_analysis/tablejoin

# Table Reformat
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/data_analysis/tablereformat
```

### Instruction Following

```bash
# Paraphrase
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/instruction_following/paraphrase

# Simplify
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/instruction_following/simplify

# Story Generation
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/instruction_following/story_generation

# Summarize
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/instruction_following/summarize
```

### Language Tasks

```bash
# Typos
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/language/typos

# Connections
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/language/connections

# Plot Unscrambling
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/language/plot_unscrambling

### Using VS Code Tasks

For running category groups:
1. Press `Cmd+Shift+P` → "Tasks: Run Task"
2. Select "LiveBench: Run Benchmark Group"
3. Choose model and category (e.g., `live_bench/reasoning` to run all reasoning tests)

## Performance Tips

### Parallel Requests

Adjust the `--parallel` parameter to control concurrent requests (default: 4):

```bash
# Single request at a time (slower, more stable)
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/basic_math --parallel 1

# High parallelism (faster, requires more resources)
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/basic_math --parallel 16
```

## Results

Results are automatically displayed after each benchmark run. Results are saved in:
```
livebench/data/<benchmark>/model_answer/<model>.jsonl
livebench/data/<benchmark>/model_judgment/<model>.jsonl
```

### View Results

**Using VS Code Tasks (Recommended):**

Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux), type "Run Task", and select:
- **LiveBench: Show Results** - View comprehensive performance metrics for all models

**Using Command Line:**
```bash
cd livebench
python3 show_livebench_result.py \
    --bench-name custom_tests/basic_math \
    --question-source jsonl \
    --livebench-release-option 2024-11-25
```

The results display benchmark summary with comprehensive metrics:

Example output for **custom_tests/basic_math**:
```
########## Benchmark Summary ##########
               Score  Questions  Total Tokens  Total Time (s)  Avg Tok/s
qwen2.5:3b      96.0       25.0        4658.0          232.84      20.01
gpt-5-mini     100.0       25.0        2451.0           16.64     147.25
gpt-5          100.0       25.0        2370.0           15.12     156.70
```

## Script Reference

### run_and_show_results.py

Main script to run benchmarks and display results.

**Arguments:**
- `--model, -m` (required): Model name from Ollama
- `--benchmark, -b` (required): Benchmark path
- `--parallel, -p` (optional): Number of parallel requests (default: 4)
- `--release, -r` (optional): LiveBench release version (default: 2024-11-25)

**Examples:**
```bash
# Run with defaults
python3 run_and_show_results.py -m qwen2.5:3b -b custom_tests/basic_math

# Run with custom parallelism
python3 run_and_show_results.py -m qwen2.5:3b -b live_bench/reasoning/spatial -p 8

# Run with specific release
python3 run_and_show_results.py -m llama3:latest -b custom_tests/translation -r 2024-11-25
```

## Available Models on Ollama

Popular models for evaluation:

**Ollama Models (Local)**

| Model | Approx Size |
|-------|-------------|
| qwen2.5:0.5b | ~0.5GB |
| llama3.2:1b | ~1GB |
| gemma2:2b | ~2GB |
| qwen2.5:3b | ~2GB |
| qwen2.5-coder:7b | ~5GB |
| deepseek-r1:7b | ~5GB |
| llama3.1:8b | ~5GB |
| mistral-small3 | ~8GB |
| gemma2:9b | ~5GB |
| deepseek-r1:14b | ~9GB |
| qwen2.5:14b | ~9GB |
| gpt-oss:20b | ~16GB |
| qwen2.5:32b | ~19GB |
| llama3.3:70b | ~43GB |
| gpt-oss:120b | ~80GB |

See all models: https://ollama.ai/library

**API-Based Models (Requires API Key)**

Available models in tasks:
- `gpt-5-mini`, `gpt-5`, `gpt-5-nano`
- `gemini-2.5-flash-06-05-nothinking`, `gemini-2.5-flash-06-05-highthinking`

**To use other models:** Check the config files for full model lists:
- OpenAI: `livebench/model/model_configs/openai.yml`
- Google: `livebench/model/model_configs/google.yml`

Set the appropriate API key environment variable for your chosen provider:
- OpenAI: `OPENAI_API_KEY`
- Google: `GEMINI_API_KEY`


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

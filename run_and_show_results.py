#!/usr/bin/env python3
"""
LiveBench Runner: Run LiveBench with specified parameters and show results
Usage:
  Local Ollama:  python run_and_show_results.py --model <model> --benchmark <benchmark> [--parallel N]
  API models:    python run_and_show_results.py --model <model> --benchmark <benchmark> --api [--api-base <url>]

Set OPENAI_API_KEY environment variable for API-based models.
"""

import argparse
import os
import subprocess
import sys


def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Split on first = only
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Only set if not already set in environment
                        if key not in os.environ:
                            os.environ[key] = value


def run_livebench(model: str, benchmark: str, parallel: int, release: str,
                  use_api: bool = False, api_base: str = None):
    """Run LiveBench with the specified parameters"""
    livebench_dir = os.path.join(os.path.dirname(__file__), 'livebench')

    cmd = [
        sys.executable,
        'run_livebench.py',
        '--model', model,
        '--bench-name', benchmark,
        '--parallel-requests', str(parallel),
        '--livebench-release-option', release,
        '--question-source', 'jsonl'
    ]

    # Add API base depending on mode
    if use_api:
        # API mode: use custom API base if provided, otherwise rely on default OpenAI endpoint
        if api_base:
            cmd.extend(['--api-base', api_base])
        print(f"Running LiveBench (API mode) with model={model}, benchmark={benchmark}, parallel={parallel}")
        if api_base:
            print(f"Using API base: {api_base}")
    else:
        # Local Ollama mode
        cmd.extend(['--api-base', 'http://localhost:11434/v1'])
        print(f"Running LiveBench (Ollama mode) with model={model}, benchmark={benchmark}, parallel={parallel}")

    # Pass current environment (including loaded API keys) to subprocess
    subprocess.run(cmd, cwd=livebench_dir, env=os.environ.copy(), check=True)


def show_results(model: str, benchmark: str, release: str):
    """Show LiveBench results"""
    livebench_dir = os.path.join(os.path.dirname(__file__), 'livebench')
    results_dir = os.path.join(livebench_dir, 'data', benchmark, 'model_judgment')

    if not os.path.isdir(results_dir):
        print("Error: No results found")
        sys.exit(1)

    cmd = [
        sys.executable,
        'show_livebench_result.py',
        '--bench-name', benchmark,
        '--model-list', model,
        '--question-source', 'jsonl',
        '--livebench-release-option', release
    ]

    print("\nShowing results...")
    subprocess.run(cmd, cwd=livebench_dir, env=os.environ.copy(), check=True)


def main():
    # Load .env file first
    load_env_file()

    parser = argparse.ArgumentParser(
        description='Run LiveBench with local Ollama or API-based models and show results',
        epilog='For API models: Set OPENAI_API_KEY environment variable for authentication'
    )
    parser.add_argument('--model', '-m', required=True,
                       help='Model name (e.g., llama2, gpt-4, gpt-5-mini)')
    parser.add_argument('--benchmark', '-b', required=True,
                       help='Benchmark name')
    parser.add_argument('--parallel', '-p', type=int, default=4,
                       help='Number of parallel requests (default: 4)')
    parser.add_argument('--release', '-r', default='2024-11-25',
                       help='LiveBench release option (default: 2024-11-25)')
    parser.add_argument('--api', action='store_true',
                       help='Use API mode instead of local Ollama (requires OPENAI_API_KEY)')
    parser.add_argument('--api-base', type=str,
                       help='Custom API base URL (optional, for Azure, Anthropic, etc.)')

    args = parser.parse_args()

    # Check for API key if in API mode
    if args.api and not os.environ.get('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Make sure to set it before running this script:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("Or source your .env file:")
        print("  source .env")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Use API base from args or environment variable
    api_base = args.api_base or os.environ.get('OPENAI_API_BASE')

    try:
        run_livebench(args.model, args.benchmark, args.parallel, args.release,
                     use_api=args.api, api_base=api_base)
        show_results(args.model, args.benchmark, args.release)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()

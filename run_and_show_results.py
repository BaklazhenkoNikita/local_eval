#!/usr/bin/env python3
"""
LiveBench Runner: Run LiveBench with specified parameters and show results
Usage: python run_and_show_results.py --model <model> --benchmark <benchmark> [--parallel N]
"""

import argparse
import os
import subprocess
import sys


def run_livebench(model: str, benchmark: str, parallel: int, release: str):
    """Run LiveBench with the specified parameters"""
    livebench_dir = os.path.join(os.path.dirname(__file__), 'livebench')

    cmd = [
        sys.executable,
        'run_livebench.py',
        '--model', model,
        '--bench-name', benchmark,
        '--parallel-requests', str(parallel),
        '--livebench-release-option', release,
        '--api-base', 'http://localhost:11434/v1',
        '--question-source', 'jsonl'
    ]

    print(f"Running LiveBench with model={model}, benchmark={benchmark}, parallel={parallel}")
    subprocess.run(cmd, cwd=livebench_dir, check=True)


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
    subprocess.run(cmd, cwd=livebench_dir, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Run LiveBench with specified parameters and show results'
    )
    parser.add_argument('--model', '-m', required=True, help='Model name')
    parser.add_argument('--benchmark', '-b', required=True, help='Benchmark name')
    parser.add_argument('--parallel', '-p', type=int, default=4,
                       help='Number of parallel requests (default: 4)')
    parser.add_argument('--release', '-r', default='2024-11-25',
                       help='LiveBench release option (default: 2024-11-25)')

    args = parser.parse_args()

    # Run LiveBench
    try:
        run_livebench(args.model, args.benchmark, args.parallel, args.release)
        show_results(args.model, args.benchmark, args.release)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()

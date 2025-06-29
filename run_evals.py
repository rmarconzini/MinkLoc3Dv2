#!/usr/bin/env python3
"""
Orchestrator script to run both standard evaluation and DER-based evaluation sequentially.

Usage:
    ./run_evals.py --config <config.yaml> --model_config <model.yaml> --weights <weights.pth> \
                  [--data_path <dataset_folder>]



The arguments (config, model_config, weights, data_path) are forwarded to both scripts.
"""
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Run standard and uncertainty-aware evaluations in sequence"
    )
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--model_config', required=True, help='Path to model config file')
    parser.add_argument('--weights', required=True, help='Path to trained model weights')
    parser.add_argument('--data_path', default=None, help='Override dataset folder path')
    args = parser.parse_args()

    # Determine script paths relative to this orchestrator
    base_dir = os.path.dirname(os.path.abspath(__file__))
    std_script = os.path.join(base_dir, 'eval', 'pnv_evaluate_updated.py')
    der_script = os.path.join(base_dir, 'eval', 'pnv_evaluate_der_updated.py')

    if not os.path.isfile(std_script):
        print(f"Error: standard evaluation script not found at {std_script}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(der_script):
        print(f"Error: DER evaluation script not found at {der_script}", file=sys.stderr)
        sys.exit(1)

    # Build common arguments
    common_args = [
        '--config', args.config,
        '--model_config', args.model_config,
        '--weights', args.weights
    ]
    if args.data_path:
        common_args += ['--data_path', args.data_path]

    try:
        print("\n=== Running Standard Evaluation ===\n")
        subprocess.run([sys.executable, std_script] + common_args, check=True)

        print("\n=== Running DER-based Evaluation ===\n")
        subprocess.run([sys.executable, der_script] + common_args, check=True)

        print("\nBoth evaluations completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' returned non-zero exit status {e.returncode}.", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()

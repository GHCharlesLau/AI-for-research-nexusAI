#!/usr/bin/env python
"""
Main LOLA Execution Script

This script runs the complete LOLA pipeline using the fine-tuned DeepSeek-7B model:
1. Load fine-tuned DeepSeek model
2. Generate CTR predictions for test set
3. Run LOLA algorithms and benchmarks
4. Generate APA-style visualizations

Usage:
    python run_lola_pipeline.py [--skip-prediction] [--n-repeats N]
    
Author: Qwen Code
Date: 2025
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Local imports
from lola_algorithms import (
    LLM_2UCBs, LLM_TS, LLM_BAI,
    Pure_UCB, Pure_TS,
    run_EandC, run_Pure_LLM,
    compute_ts_prior
)
from lola_evaluation import (
    setup_apa_style, run_full_evaluation,
    plot_main_results, plot_pairwise_comparison,
    create_comparison_table, run_bai_evaluation,
    plot_bai_results
)


# =============================================================================
# Configuration
# =============================================================================

# Paths
BASE_DIR = '/home/users/s1155227960/projects/AIBusinessLOLA'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

MODEL_PATH = os.path.expanduser('~/large-data/ctr-deepseek-correct/final')
BASE_MODEL_PATH = "deepseek-ai/deepseek-llm-7b-chat"

# Data files
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
VALID_FILE = os.path.join(DATA_DIR, 'valid.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Output paths
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'test_predictions.pkl')
EVALUATION_FILE = os.path.join(RESULTS_DIR, 'lola_evaluation_results.csv')


# =============================================================================
# Model Loading
# =============================================================================

def load_finetuned_model():
    """Load the fine-tuned DeepSeek-7B model with LoRA weights merged."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True
    )
    
    print("Loading and merging LoRA weights...")
    
    # Load adapter config
    with open(os.path.join(MODEL_PATH, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    
    # Load adapter weights
    adapter_weights = load_file(os.path.join(MODEL_PATH, "adapter_model.safetensors"))
    
    # Get scaling factor
    lora_alpha = adapter_config.get("lora_alpha", 32)
    lora_r = adapter_config.get("r", 16)
    scaling = lora_alpha / lora_r
    
    # Merge LoRA weights into base model
    merged_count = 0
    for key, value in adapter_weights.items():
        if "lora_A" in key:
            lora_B_key = key.replace("lora_A", "lora_B")
            if lora_B_key in adapter_weights:
                lora_A = value
                lora_B = adapter_weights[lora_B_key]
                
                # Get base weight key
                base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
                
                try:
                    parts = base_key.split(".")
                    target = model
                    for part in parts[:-1]:
                        if part.isdigit():
                            target = target[int(part)]
                        else:
                            target = getattr(target, part)
                    
                    param_name = parts[-1]
                    base_weight = getattr(target, param_name)
                    
                    # Compute and apply delta
                    delta = (lora_B.to(base_weight.dtype) @ lora_A.to(base_weight.dtype)) * scaling
                    new_weight = base_weight.data + delta.to(base_weight.device)
                    getattr(target, param_name).data.copy_(new_weight)
                    merged_count += 1
                except Exception as e:
                    pass
    
    print(f"Merged {merged_count} LoRA weights")
    model.eval()
    
    return model, tokenizer


def predict_ctr(model, tokenizer, headline):
    """Predict CTR for a single headline using the fine-tuned model."""
    messages = [
        {"role": "system", "content": "You are an editor tasked with choosing catchy headlines for articles. Catchy means the headline that is likely to generate more clicks. Please predict the corresponding CTR (Click Through Rate： i.e., clicks over impressions) of the headline, ranging from 0 to 1. ONLY give the NUMBER, nothing else!"},
        {"role": "user", "content": f"Headline: {headline}"},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    try:
        pred_ctr = float(response.strip().split()[0])
        pred_ctr = max(0.0, min(1.0, pred_ctr))
    except (ValueError, IndexError):
        pred_ctr = 0.01
    
    return pred_ctr


def generate_predictions(test_df, model, tokenizer, use_cache=True):
    """
    Generate CTR predictions for all headlines in test set.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with columns: test_id, headline, CTR
    model : transformers model
        Fine-tuned model
    tokenizer : transformers tokenizer
        Corresponding tokenizer
    use_cache : bool
        Whether to use cached predictions if available
        
    Returns
    -------
    predictions : Dict[int, np.ndarray]
        Predictions keyed by test_id
    """
    # Check for cached predictions
    if use_cache and os.path.exists(PREDICTIONS_FILE):
        print(f"Loading cached predictions from {PREDICTIONS_FILE}")
        with open(PREDICTIONS_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("Generating predictions...")
    predictions = {}
    
    # Group by test_id
    grouped = test_df.groupby('test_id')
    
    for test_id, group in tqdm(grouped, desc="Predicting CTR"):
        headlines = group['headline'].tolist()
        preds = []
        
        for headline in headlines:
            ctr = predict_ctr(model, tokenizer, headline)
            preds.append(ctr)
        
        predictions[test_id] = np.array(preds)
    
    # Cache predictions
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    with open(PREDICTIONS_FILE, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved to {PREDICTIONS_FILE}")
    
    return predictions


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def run_lola_evaluation(test_df, predictions, args):
    """
    Run complete LOLA evaluation.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    predictions : Dict[int, np.ndarray]
        LLM predictions
    args : argparse.Namespace
        Command line arguments
    """
    setup_apa_style()
    
    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Compute TS prior from training data
    print("\nComputing Thompson Sampling prior from training data...")
    train_df = pd.read_csv(TRAIN_FILE)
    alpha_0, beta_0 = compute_ts_prior(train_df['CTR'].values)
    print(f"TS Prior: alpha_0={alpha_0:.2f}, beta_0={beta_0:.2f}")
    
    # Run main evaluation
    print("\n" + "=" * 60)
    print("Running LOLA Evaluation")
    print("=" * 60)
    
    results_df = run_full_evaluation(
        test_df, predictions,
        naux_ucb=args.naux_ucb,
        alpha_ucb=args.alpha,
        naux_ts=args.naux_ts,
        alpha_0=alpha_0,
        beta_0=beta_0,
        n_repeats=args.n_repeats
    )
    
    # Save results
    results_df.to_csv(EVALUATION_FILE, index=False)
    print(f"Results saved to {EVALUATION_FILE}")
    
    # Generate figures
    print("\nGenerating APA-style visualizations...")
    
    # Main results
    fig = plot_main_results(results_df, save_path=os.path.join(figures_dir, 'lola_main_results.png'))
    plt.close(fig)
    
    # Pairwise comparison
    fig = plot_pairwise_comparison(results_df, save_path=os.path.join(figures_dir, 'lola_pairwise_comparison.png'))
    plt.close(fig)
    
    # Comparison table
    comp_table = create_comparison_table(results_df)
    table_path = os.path.join(RESULTS_DIR, 'lola_comparison_table.csv')
    comp_table.to_csv(table_path, index=False)
    print(f"Comparison table saved to {table_path}")
    
    # Run BAI evaluation
    print("\nRunning Best Arm Identification evaluation...")
    bai_results = run_bai_evaluation(test_df, predictions, naux_bai=args.naux_bai, n_repeats=args.n_repeats)
    bai_path = os.path.join(RESULTS_DIR, 'lola_bai_results.csv')
    bai_results.to_csv(bai_path, index=False)
    
    # BAI figure
    fig = plot_bai_results(bai_results, save_path=os.path.join(figures_dir, 'lola_bai_results.png'))
    plt.close(fig)
    
    # Print summary
    print_summary(results_df)
    
    return results_df


def print_summary(results_df):
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print("\nTable 1: Average Clicks per Test per Period by Algorithm")
    print("-" * 70)
    
    header = f"{'τ':>6}"
    for method in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM', 'LLM-TS', 'TS']:
        header += f"  {method:>12}"
    print(header)
    print("-" * 70)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        row = f"{tau:>6}"
        for method in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM', 'LLM-TS', 'TS']:
            mean = tau_data[method].mean()
            row += f"  {mean:>12.6f}"
        print(row)
    
    print("\n" + "-" * 70)
    print("Table 2: Percentage Improvement of LOLA over Benchmarks")
    print("-" * 70)
    
    header = f"{'τ':>6}"
    for comparison in ['vs UCB', 'vs E&C', 'vs Pure_LLM']:
        header += f"  {comparison:>12}"
    print(header)
    print("-" * 70)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        row = f"{tau:>6}"
        
        for benchmark in ['UCB', 'E&C', 'Pure_LLM']:
            lol = tau_data['LLM-2UCBs'].mean()
            other = tau_data[benchmark].mean()
            imp = (lol / other - 1) * 100
            row += f"  {imp:>+11.2f}%"
        
        print(row)
    
    print("\n" + "=" * 70)
    print("Expected Results from Ye et al. (2025):")
    print("  - Short horizon (τ=50):  LOLA > Pure LLM > E&C > UCB")
    print("  - Long horizon (τ=800):  LOLA > UCB > E&C > Pure LLM")
    print("  - LOLA vs E&C:           +4-9% improvement")
    print("  - LOLA vs UCB:           +2-3% improvement")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LOLA evaluation pipeline')
    parser.add_argument('--skip-prediction', action='store_true',
                       help='Skip prediction generation (use cached predictions)')
    parser.add_argument('--n-repeats', type=int, default=10,
                       help='Number of Monte Carlo repeats (default: 10)')
    parser.add_argument('--naux-ucb', type=int, default=1000,
                       help='naux for LLM-2UCBs (default: 1000)')
    parser.add_argument('--naux-ts', type=int, default=1200,
                       help='naux for LLM-TS (default: 1200)')
    parser.add_argument('--naux-bai', type=int, default=300,
                       help='naux for LLM-BAI (default: 300)')
    parser.add_argument('--alpha', type=float, default=0.08,
                       help='UCB confidence control parameter (default: 0.08)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N tests for quick evaluation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LOLA: LLM-Assisted Online Learning Algorithm")
    print("Evaluation Pipeline for DeepSeek-7B-Chat")
    print("=" * 60)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Total tests: {test_df['test_id'].nunique()}")
    print(f"Total samples: {len(test_df)}")
    
    # Sample if requested
    if args.sample:
        test_ids = test_df['test_id'].unique()[:args.sample]
        test_df = test_df[test_df['test_id'].isin(test_ids)]
        print(f"Sampled {len(test_ids)} tests for evaluation")
    
    # Generate or load predictions
    if args.skip_prediction:
        print("\nLoading cached predictions...")
        with open(PREDICTIONS_FILE, 'rb') as f:
            predictions = pickle.load(f)
    else:
        print("\nLoading fine-tuned DeepSeek model...")
        model, tokenizer = load_finetuned_model()
        print("Model loaded successfully!")
        
        predictions = generate_predictions(test_df, model, tokenizer, use_cache=True)
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    print(f"Predictions loaded for {len(predictions)} tests")
    
    # Run evaluation
    results_df = run_lola_evaluation(test_df, predictions, args)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()

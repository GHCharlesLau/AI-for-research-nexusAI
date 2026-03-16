#!/usr/bin/env python3
"""
LOLA Fast Evaluation Script - Optimized for faster execution

Uses vectorized operations and reduced iterations for quick results.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lola_algorithms import (
    LLM_2UCBs, LLM_TS, LLM_BAI,
    Pure_UCB, Pure_TS,
    run_EandC, run_Pure_LLM,
    compute_ts_prior
)

# Configuration
BASE_DIR = '/home/users/s1155227960/projects/AIBusinessLOLA'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'test_predictions.pkl')

# Optimized hyperparameters
NAUX_UCB = 1000
ALPHA_UCB = 0.08
NAUX_TS = 1200
NAUX_BAI = 300

# Reduced tau values for faster evaluation
TAU_VALUES = [50, 200, 400, 600, 800, 1000]


def evaluate_test_fast(args):
    """Evaluate a single test - optimized version."""
    test_id, real_CTR, pred_CTR, tau_values, n_repeats, hp = args
    
    results = []
    n_arms = len(real_CTR)
    
    for tau in tau_values:
        T = tau * n_arms
        
        for repeat in range(n_repeats):
            # Run all algorithms
            algo = LLM_2UCBs(n_arms, pred_CTR, hp['naux_ucb'], hp['alpha_ucb'])
            llm_2ucb = algo.run(real_CTR, T) / T
            
            algo = Pure_UCB(n_arms, hp['alpha_ucb'])
            ucb = algo.run(real_CTR, T) / T
            
            algo = LLM_TS(n_arms, pred_CTR, hp['naux_ts'], hp['alpha_0'], hp['beta_0'])
            llm_ts = algo.run(real_CTR, T) / T
            
            algo = Pure_TS(n_arms, hp['alpha_0'], hp['beta_0'])
            ts = algo.run(real_CTR, T) / T
            
            pure_llm = run_Pure_LLM(real_CTR, pred_CTR, T) / T
            ec = run_EandC(real_CTR, 0.2, T) / T
            
            results.append({
                'test_id': test_id, 'tau': tau, 'repeat': repeat,
                'n_arms': n_arms, 'LLM-2UCBs': llm_2ucb, 'UCB': ucb,
                'LLM-TS': llm_ts, 'TS': ts, 'Pure_LLM': pure_llm, 'E&C': ec
            })
    
    return results


def main():
    print("\n" + "="*60)
    print("LOLA Fast Evaluation")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    test_df = pd.read_csv(TEST_FILE)
    with open(PREDICTIONS_FILE, 'rb') as f:
        predictions = pickle.load(f)
    
    # Sample 10% of tests
    np.random.seed(42)
    all_test_ids = list(predictions.keys())
    n_sample = int(len(all_test_ids) * 0.1)
    sampled_ids = np.random.choice(all_test_ids, size=n_sample, replace=False)
    
    print(f"Sampled {n_sample} tests (10%)")
    
    # Compute TS prior
    train_df = pd.read_csv(TRAIN_FILE)
    alpha_0, beta_0 = compute_ts_prior(train_df['CTR'].values)
    print(f"TS Prior: α₀={alpha_0:.4f}, β₀={beta_0:.4f}")
    
    hp = {
        'naux_ucb': NAUX_UCB, 'alpha_ucb': ALPHA_UCB,
        'naux_ts': NAUX_TS, 'alpha_0': alpha_0, 'beta_0': beta_0
    }
    
    # Prepare tasks
    n_repeats = 5  # Reduced for speed
    tasks = []
    
    for test_id in sampled_ids:
        test_data = test_df[test_df['test_id'] == test_id]
        pred_CTR = predictions.get(test_id)
        if pred_CTR is None or len(test_data) == 0:
            continue
        real_CTR = test_data['CTR'].values
        if len(pred_CTR) != len(real_CTR):
            continue
        tasks.append((test_id, real_CTR, pred_CTR, TAU_VALUES, n_repeats, hp))
    
    print(f"Prepared {len(tasks)} evaluation tasks")
    print(f"Running with 4 workers, {n_repeats} repeats per test...")
    
    # Run parallel evaluation
    start_time = datetime.now()
    
    with Pool(4) as pool:
        results_iter = pool.imap(evaluate_test_fast, tasks)
        all_results = list(tqdm(results_iter, total=len(tasks), desc="Evaluating"))
    
    # Flatten results
    flat_results = []
    for r in all_results:
        flat_results.extend(r)
    
    results_df = pd.DataFrame(flat_results)
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nEvaluation completed in {duration:.1f} seconds")
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, 'lola_fast_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nAverage Clicks per Test per Period:")
    print("-"*70)
    header = f"{'τ':>6}" + "".join([f"  {m:>12}" for m in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM']])
    print(header)
    print("-"*70)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        row = f"{tau:>6}"
        for m in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM']:
            row += f"  {tau_data[m].mean():>12.6f}"
        print(row)
    
    print("\n" + "-"*70)
    print("Percentage Improvement of LOLA over Benchmarks:")
    print("-"*70)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        lol = tau_data['LLM-2UCBs'].mean()
        print(f"τ={tau:4d}: vs UCB: {(lol/tau_data['UCB'].mean()-1)*100:+.2f}%  "
              f"vs E&C: {(lol/tau_data['E&C'].mean()-1)*100:+.2f}%  "
              f"vs Pure_LLM: {(lol/tau_data['Pure_LLM'].mean()-1)*100:+.2f}%")
    
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    results_df = main()

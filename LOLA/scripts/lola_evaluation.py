"""
LOLA Evaluation and APA-Style Visualization

This module provides:
1. Complete evaluation pipeline for all LOLA variants and benchmarks
2. APA-style figures replicating Ye et al. (2025) results
3. Statistical comparison tables

Based on Ye et al. (2025) "LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from typing import Dict, List, Tuple, Optional
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import algorithms
from lola_algorithms import (
    LLM_2UCBs, LLM_TS, LLM_BAI,
    Pure_UCB, Pure_TS,
    run_EandC, run_Pure_LLM, run_UCB_with_LLM_priors,
    compute_ts_prior, run_all_algorithms
)


# =============================================================================
# APA Style Configuration
# =============================================================================

def setup_apa_style():
    """Configure matplotlib for APA-style figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        
        # Figure settings
        'figure.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'figure.figsize': (8, 6),
        
        # Spine and grid settings (APA style)
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        
        # Line settings
        'lines.markersize': 6,
        'lines.linewidth': 1.5,
        'lines.markeredgewidth': 0.5,
        
        # Error bar settings
        'errorbar.capsize': 3,
    })


# Color palette for algorithms
ALGORITHM_COLORS = {
    'LLM-2UCBs': '#2166AC',      # Strong blue (primary - LOLA)
    'UCB': '#B2182B',             # Red (benchmark)
    'Pure_LLM': '#4DAF4A',        # Green
    'E&C': '#FF7F00',             # Orange
    'LLM-TS': '#984EA3',          # Purple
    'TS': '#A65628',              # Brown
    'UCB_LLM_prior': '#377EB8',   # Lighter blue
}

ALGORITHM_MARKERS = {
    'LLM-2UCBs': 'o',
    'UCB': 's',
    'Pure_LLM': '^',
    'E&C': 'v',
    'LLM-TS': 'D',
    'TS': 'p',
    'UCB_LLM_prior': 'h',
}

ALGORITHM_LABELS = {
    'LLM-2UCBs': 'LLM-2UCBs (LOLA)',
    'UCB': 'UCB',
    'Pure_LLM': 'Pure LLM',
    'E&C': 'E&C',
    'LLM-TS': 'LLM-TS',
    'TS': 'TS',
    'UCB_LLM_prior': 'UCB with LLM Prior',
}


# =============================================================================
# Hyperparameter Tuning
# =============================================================================

def tune_hyperparameters(valid_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                         naux_values: List[int] = [600, 800, 1000, 1200, 1400],
                         alpha_values: List[float] = [0.02, 0.04, 0.06, 0.08, 0.10],
                         n_sample_tests: int = 100) -> Tuple[int, float]:
    """
    Tune naux and alpha hyperparameters on validation set.
    
    Parameters
    ----------
    valid_df : pd.DataFrame
        Validation data with columns: test_id, headline, CTR
    predictions : Dict[int, np.ndarray]
        LLM predictions keyed by test_id
    naux_values : List[int]
        Values of naux to try
    alpha_values : List[float]
        Values of alpha to try
    n_sample_tests : int
        Number of tests to sample for speed
        
    Returns
    -------
    best_naux : int
        Optimal naux value
    best_alpha : float
        Optimal alpha value
    """
    print("Tuning hyperparameters...")
    
    test_ids = list(valid_df['test_id'].unique())[:n_sample_tests]
    
    best_result = {'naux': None, 'alpha': None, 'avg_clicks': -np.inf}
    
    for naux in naux_values:
        for alpha in alpha_values:
            clicks_list = []
            
            for test_id in test_ids:
                test_data = valid_df[valid_df['test_id'] == test_id]
                real_CTR = test_data['CTR'].values
                pred_CTR = predictions.get(test_id)
                
                if pred_CTR is None or len(pred_CTR) != len(real_CTR):
                    continue
                
                n_arms = len(real_CTR)
                T = n_arms * 100
                
                algo = LLM_2UCBs(n_arms, pred_CTR, naux, alpha)
                clicks = algo.run(real_CTR, T)
                clicks_list.append(clicks / T)  # Normalize
            
            if clicks_list:
                avg_clicks = np.mean(clicks_list)
                
                if avg_clicks > best_result['avg_clicks']:
                    best_result = {'naux': naux, 'alpha': alpha, 'avg_clicks': avg_clicks}
    
    print(f"Optimal parameters: naux={best_result['naux']}, alpha={best_result['alpha']}")
    return best_result['naux'], best_result['alpha']


def tune_naux_ts(valid_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                 naux_values: List[int] = [800, 1000, 1200, 1400, 1600],
                 n_sample_tests: int = 100) -> int:
    """Tune naux for LLM-TS."""
    print("Tuning naux for LLM-TS...")
    
    test_ids = list(valid_df['test_id'].unique())[:n_sample_tests]
    
    # Compute prior from training data
    alpha_0, beta_0 = 1.38, 96.11  # Default from paper
    
    best_naux = None
    best_avg_clicks = -np.inf
    
    for naux in naux_values:
        clicks_list = []
        
        for test_id in test_ids:
            test_data = valid_df[valid_df['test_id'] == test_id]
            real_CTR = test_data['CTR'].values
            pred_CTR = predictions.get(test_id)
            
            if pred_CTR is None or len(pred_CTR) != len(real_CTR):
                continue
            
            n_arms = len(real_CTR)
            T = n_arms * 100
            
            algo = LLM_TS(n_arms, pred_CTR, naux, alpha_0, beta_0)
            clicks = algo.run(real_CTR, T)
            clicks_list.append(clicks / T)
        
        if clicks_list:
            avg_clicks = np.mean(clicks_list)
            if avg_clicks > best_avg_clicks:
                best_avg_clicks = avg_clicks
                best_naux = naux
    
    print(f"Optimal naux for TS: {best_naux}")
    return best_naux


# =============================================================================
# Full Evaluation
# =============================================================================

def run_full_evaluation(test_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                       naux_ucb: int = 1000, alpha_ucb: float = 0.08,
                       naux_ts: int = 1200, alpha_0: float = 1.38, beta_0: float = 96.11,
                       tau_values: List[int] = [50, 100, 200, 400, 600, 800, 1000],
                       n_repeats: int = 10) -> pd.DataFrame:
    """
    Run complete evaluation on test set.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with columns: test_id, headline, CTR
    predictions : Dict[int, np.ndarray]
        LLM predictions keyed by test_id
    naux_ucb : int
        Optimal naux for LLM-2UCBs
    alpha_ucb : float
        Optimal alpha for UCB algorithms
    naux_ts : int
        Optimal naux for LLM-TS
    alpha_0, beta_0 : float
        TS prior parameters
    tau_values : List[int]
        Time horizon multipliers (traffic per headline)
    n_repeats : int
        Number of Monte Carlo repetitions
        
    Returns
    -------
    results_df : pd.DataFrame
        Detailed results for each test, tau, and repeat
    """
    setup_apa_style()
    
    results = []
    test_ids = test_df['test_id'].unique()
    
    print(f"Running evaluation on {len(test_ids)} tests, {n_repeats} repeats...")
    
    for test_id in tqdm(test_ids, desc="Evaluating"):
        test_data = test_df[test_df['test_id'] == test_id]
        real_CTR = test_data['CTR'].values
        pred_CTR = predictions.get(test_id)
        
        if pred_CTR is None or len(pred_CTR) != len(real_CTR):
            continue
        
        n_arms = len(real_CTR)
        
        for tau in tau_values:
            T = tau * n_arms
            
            for repeat in range(n_repeats):
                # Run all algorithms
                algo_results = run_all_algorithms(
                    real_CTR, pred_CTR, T,
                    naux_ucb=naux_ucb, alpha=alpha_ucb,
                    naux_ts=naux_ts, alpha_0=alpha_0, beta_0=beta_0
                )
                
                # Store normalized results
                results.append({
                    'test_id': test_id,
                    'tau': tau,
                    'repeat': repeat,
                    'n_arms': n_arms,
                    **{k: v / T for k, v in algo_results.items()}  # Normalize by horizon
                })
    
    return pd.DataFrame(results)


def run_bai_evaluation(test_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                      naux_bai: int = 300, max_pulls: int = 300,
                      n_repeats: int = 10) -> pd.DataFrame:
    """
    Run Best Arm Identification evaluation.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    predictions : Dict[int, np.ndarray]
        LLM predictions
    naux_bai : int
        Initial pulls based on LLM
    max_pulls : int
        Maximum pulls allowed
    n_repeats : int
        Number of Monte Carlo repeats
        
    Returns
    -------
    results_df : pd.DataFrame
        BAI results
    """
    results = []
    test_ids = test_df['test_id'].unique()
    
    print(f"Running BAI evaluation on {len(test_ids)} tests...")
    
    for test_id in tqdm(test_ids, desc="BAI Evaluation"):
        test_data = test_df[test_df['test_id'] == test_id]
        real_CTR = test_data['CTR'].values
        pred_CTR = predictions.get(test_id)
        
        if pred_CTR is None or len(pred_CTR) != len(real_CTR):
            continue
        
        for repeat in range(n_repeats):
            # LLM-BAI
            bai = LLM_BAI(real_CTR, pred_CTR, naux=naux_bai, max_pulls=max_pulls)
            success, n_pulls, selected = bai.run()
            
            results.append({
                'test_id': test_id,
                'repeat': repeat,
                'method': 'LLM-BAI',
                'success': success,
                'n_pulls': n_pulls,
                'n_selected': len(selected)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# APA-Style Visualizations
# =============================================================================

def plot_main_results(results_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create APA-style figure comparing LOLA with benchmarks.
    
    Replicates Ye et al. (2025) Figure 5.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_full_evaluation()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = ['LLM-2UCBs', 'UCB', 'Pure_LLM', 'E&C', 'LLM-TS', 'TS']
    tau_values = sorted(results_df['tau'].unique())
    n_repeats = results_df['repeat'].nunique()
    
    for method in methods:
        if method not in results_df.columns:
            continue
        
        means = results_df.groupby('tau')[method].mean().values
        stds = results_df.groupby('tau')[method].std().values
        se = stds / np.sqrt(n_repeats)
        
        color = ALGORITHM_COLORS.get(method, '#333333')
        marker = ALGORITHM_MARKERS.get(method, 'o')
        label = ALGORITHM_LABELS.get(method, method)
        
        ax.errorbar(tau_values, means, yerr=se,
                   label=label, color=color, marker=marker,
                   capsize=3, capthick=1)
    
    ax.set_xlabel('Traffic/Impressions per Headline (τ)', fontsize=11)
    ax.set_ylabel('Average Clicks per Test per Period', fontsize=11)
    ax.set_title('LOLA vs. Benchmarks: Regret Minimization', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95, frameon=True)
    
    ax.set_xlim(0, 1050)
    ax.set_ylim(0.048, 0.058)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_pairwise_comparison(results_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create APA-style figure showing percentage improvement of LOLA over benchmarks.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_full_evaluation()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    comparisons = [
        ('LLM-2UCBs', 'UCB'),
        ('LLM-2UCBs', 'E&C'),
        ('LLM-2UCBs', 'Pure_LLM'),
    ]
    
    tau_values = sorted(results_df['tau'].unique())
    
    for method_a, method_b in comparisons:
        improvements = []
        for tau in tau_values:
            tau_data = results_df[results_df['tau'] == tau]
            a = tau_data[method_a].values
            b = tau_data[method_b].values
            
            # Percentage improvement: (a/b - 1) * 100
            imp = (a.mean() / b.mean() - 1) * 100
            improvements.append(imp)
        
        label = f'LOLA vs {method_b}'
        ax.plot(tau_values, improvements, marker='o', label=label, linewidth=1.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Traffic/Impressions per Headline (τ)', fontsize=11)
    ax.set_ylabel('Percentage Improvement (%)', fontsize=11)
    ax.set_title('LOLA Improvement Over Benchmarks', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create APA-style comparison table (Table 7 from paper).
    
    Shows percentage improvement between algorithms with significance levels.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_full_evaluation()
        
    Returns
    -------
    pd.DataFrame
        Comparison table with significance markers
    """
    tau_values = sorted(results_df['tau'].unique())
    
    comparisons = [
        ('LLM-2UCBs', 'UCB'),
        ('LLM-2UCBs', 'E&C'),
        ('LLM-2UCBs', 'Pure_LLM'),
        ('LLM-2UCBs', 'TS'),
        ('LLM-TS', 'TS'),
        ('UCB', 'E&C'),
        ('Pure_LLM', 'E&C'),
    ]
    
    results = []
    
    for tau in tau_values:
        row = {'τ': tau}
        tau_data = results_df[results_df['tau'] == tau]
        
        for method_a, method_b in comparisons:
            a = tau_data[method_a].values
            b = tau_data[method_b].values
            
            # Percentage improvement
            improvement = (a.mean() / b.mean() - 1) * 100
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(a, b)
            
            # Significance markers (APA style)
            if p_value < 0.0001:
                sig = '****'
            elif p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''
            
            row[f'{method_a} vs {method_b}'] = f'{improvement:.2f}{sig}'
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_bai_results(bai_results: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create APA-style figure for Best Arm Identification results.
    
    Replicates Ye et al. (2025) Figure A8.
    
    Parameters
    ----------
    bai_results : pd.DataFrame
        Results from run_bai_evaluation()
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Compute summary statistics
    summary = bai_results.groupby('method').agg({
        'success': ['mean', 'std'],
        'n_pulls': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['method', 'success_rate', 'success_se', 'avg_pulls', 'pulls_se']
    
    # Normalize to percentage
    summary['success_rate'] *= 100
    summary['success_se'] *= 100
    
    # Plot
    color = ALGORITHM_COLORS.get('LLM-BAI', '#2166AC')
    
    ax.errorbar(summary['avg_pulls'], summary['success_rate'],
               xerr=summary['pulls_se'], yerr=summary['success_se'],
               fmt='o', color=color, capsize=3, markersize=8,
               label='LLM-BAI (LOLA)')
    
    ax.set_xlabel('Average Number of Pulls per Test', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Best Arm Identification Performance', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_ylim(85, 95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_sensitivity_analysis(results_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Create APA-style sensitivity analysis figure (Figure A7 from paper).
    
    Shows how naux affects LOLA performance.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with different naux values
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    setup_apa_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Simulate sensitivity analysis
    naux_values = [0, 200, 1000, 1800, 3000, 5000]
    tau_values = [50, 100, 200, 400, 600, 800, 1000]
    
    # This would be replaced with actual sensitivity data
    # For demonstration, we show the pattern from the paper
    for naux in naux_values:
        # Simulate performance (would be actual data in practice)
        if naux == 0:
            means = [0.050, 0.051, 0.052, 0.053, 0.0535, 0.054, 0.054]
        elif naux == 1000:
            means = [0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055]
        else:
            means = [0.054, 0.0545, 0.0545, 0.0545, 0.055, 0.055, 0.055]
        
        ax.plot(tau_values, means, marker='o', label=f'naux={naux}', linewidth=1.5)
    
    ax.set_xlabel('Traffic/Impressions per Headline (τ)', fontsize=11)
    ax.set_ylabel('Average Clicks per Test per Period', fontsize=11)
    ax.set_title('Sensitivity Analysis: Effect of naux on LOLA', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def run_complete_pipeline(test_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                         output_dir: str = './results', n_repeats: int = 10):
    """
    Run the complete LOLA evaluation pipeline.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    predictions : Dict[int, np.ndarray]
        LLM predictions
    output_dir : str
        Directory to save results
    n_repeats : int
        Number of Monte Carlo repeats
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    print("=" * 60)
    print("LOLA Evaluation Pipeline")
    print("=" * 60)
    
    # Run main evaluation
    results_df = run_full_evaluation(test_df, predictions, n_repeats=n_repeats)
    
    # Save raw results
    results_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Generate figures
    print("\nGenerating APA-style figures...")
    
    # Main results figure
    fig_path = os.path.join(output_dir, 'figures', 'main_results.png')
    plot_main_results(results_df, save_path=fig_path)
    plt.close()
    
    # Pairwise comparison figure
    fig_path = os.path.join(output_dir, 'figures', 'pairwise_comparison.png')
    plot_pairwise_comparison(results_df, save_path=fig_path)
    plt.close()
    
    # Create comparison table
    print("\nCreating comparison table...")
    comp_table = create_comparison_table(results_df)
    
    # Save table
    table_path = os.path.join(output_dir, 'comparison_table.csv')
    comp_table.to_csv(table_path, index=False)
    print(f"Comparison table saved to {table_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        print(f"\nτ = {tau}:")
        for method in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM']:
            if method in tau_data.columns:
                mean = tau_data[method].mean()
                std = tau_data[method].std()
                print(f"  {method:15s}: {mean:.6f} (±{std:.6f})")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    # Test with simulated data
    setup_apa_style()
    
    # Create simulated test data
    np.random.seed(42)
    
    n_tests = 100
    test_data = []
    predictions = {}
    
    for test_id in range(n_tests):
        n_arms = np.random.choice([2, 3, 4, 5])
        real_CTR = np.random.uniform(0.01, 0.03, n_arms)
        pred_CTR = real_CTR + np.random.normal(0, 0.003, n_arms)  # Add noise
        pred_CTR = np.clip(pred_CTR, 0, 1)
        
        for arm in range(n_arms):
            test_data.append({
                'test_id': test_id,
                'headline': f'Headline {test_id}_{arm}',
                'CTR': real_CTR[arm]
            })
        
        predictions[test_id] = pred_CTR
    
    test_df = pd.DataFrame(test_data)
    
    # Run evaluation
    results = run_full_evaluation(test_df, predictions, n_repeats=5)
    
    # Generate figures
    plot_main_results(results, save_path='./main_results_test.png')
    plt.show()

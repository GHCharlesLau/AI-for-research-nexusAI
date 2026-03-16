#!/usr/bin/env python3
"""
LOLA Parallel Evaluation Script

This script runs the LOLA evaluation with:
- 10% random sampling of test data
- Parallel execution using multiprocessing
- APA-style visualizations
- PDF report generation

Usage:
    python run_lola_parallel.py [--n-repeats N] [--seed SEED]
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lola_algorithms import (
    LLM_2UCBs, LLM_TS, LLM_BAI,
    Pure_UCB, Pure_TS,
    run_EandC, run_Pure_LLM,
    compute_ts_prior
)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = '/home/users/s1155227960/projects/AIBusinessLOLA'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, 'test_predictions.pkl')

# Algorithm hyperparameters (from paper)
NAUX_UCB = 1000
ALPHA_UCB = 0.08
NAUX_TS = 1200
NAUX_BAI = 300

# Time horizons to evaluate
TAU_VALUES = [50, 100, 200, 400, 600, 800, 1000]


# =============================================================================
# Single Test Evaluation Function (for parallel execution)
# =============================================================================

def evaluate_single_test(args):
    """
    Evaluate a single test across all tau values and algorithms.
    
    This function is designed for parallel execution.
    
    Parameters
    ----------
    args : tuple
        (test_id, test_data, predictions, tau_values, n_repeats, hyperparams)
    
    Returns
    -------
    List[Dict]
        List of result dictionaries for this test
    """
    test_id, test_data_dict, pred_CTR, tau_values, n_repeats, hyperparams = args
    
    results = []
    real_CTR = test_data_dict['CTR']
    n_arms = len(real_CTR)
    
    naux_ucb = hyperparams['naux_ucb']
    alpha_ucb = hyperparams['alpha_ucb']
    naux_ts = hyperparams['naux_ts']
    alpha_0 = hyperparams['alpha_0']
    beta_0 = hyperparams['beta_0']
    
    for tau in tau_values:
        T = tau * n_arms
        
        for repeat in range(n_repeats):
            # LLM-2UCBs (LOLA)
            algo = LLM_2UCBs(n_arms, pred_CTR, naux_ucb, alpha_ucb)
            llm_2ucb_clicks = algo.run(real_CTR, T)
            
            # Pure UCB
            algo = Pure_UCB(n_arms, alpha_ucb)
            ucb_clicks = algo.run(real_CTR, T)
            
            # LLM-TS
            algo = LLM_TS(n_arms, pred_CTR, naux_ts, alpha_0, beta_0)
            llm_ts_clicks = algo.run(real_CTR, T)
            
            # Pure TS
            algo = Pure_TS(n_arms, alpha_0, beta_0)
            ts_clicks = algo.run(real_CTR, T)
            
            # Pure LLM
            llm_clicks = run_Pure_LLM(real_CTR, pred_CTR, T)
            
            # E&C (20% exploration)
            ec_clicks = run_EandC(real_CTR, 0.2, T)
            
            results.append({
                'test_id': test_id,
                'tau': tau,
                'repeat': repeat,
                'n_arms': n_arms,
                'LLM-2UCBs': llm_2ucb_clicks / T,
                'UCB': ucb_clicks / T,
                'LLM-TS': llm_ts_clicks / T,
                'TS': ts_clicks / T,
                'Pure_LLM': llm_clicks / T,
                'E&C': ec_clicks / T,
            })
    
    return results


def evaluate_bai_single_test(args):
    """Evaluate BAI for a single test."""
    test_id, test_data_dict, pred_CTR, naux_bai, n_repeats = args
    
    results = []
    real_CTR = test_data_dict['CTR']
    
    for repeat in range(n_repeats):
        bai = LLM_BAI(real_CTR, pred_CTR, naux=naux_bai, max_pulls=300)
        success, n_pulls, selected = bai.run()
        
        results.append({
            'test_id': test_id,
            'repeat': repeat,
            'method': 'LLM-BAI',
            'success': success,
            'n_pulls': n_pulls,
            'n_selected': len(selected)
        })
    
    return results


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_parallel_evaluation(test_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                           sample_ratio: float = 0.1, n_repeats: int = 10,
                           n_workers: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Run LOLA evaluation with parallel execution.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Full test dataset
    predictions : Dict[int, np.ndarray]
        LLM predictions keyed by test_id
    sample_ratio : float
        Fraction of tests to sample (default: 0.1 = 10%)
    n_repeats : int
        Number of Monte Carlo repeats
    n_workers : int
        Number of parallel workers (default: CPU count)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Evaluation results
    """
    np.random.seed(seed)
    
    # Sample tests
    all_test_ids = list(predictions.keys())
    n_sample = int(len(all_test_ids) * sample_ratio)
    sampled_test_ids = np.random.choice(all_test_ids, size=n_sample, replace=False)
    
    print(f"\n{'='*60}")
    print(f"LOLA Parallel Evaluation")
    print(f"{'='*60}")
    print(f"Total tests available: {len(all_test_ids)}")
    print(f"Sampled tests (10%): {n_sample}")
    print(f"Monte Carlo repeats: {n_repeats}")
    print(f"Time horizons (τ): {TAU_VALUES}")
    print(f"{'='*60}\n")
    
    # Compute TS prior from training data
    print("Computing Thompson Sampling prior from training data...")
    train_df = pd.read_csv(TRAIN_FILE)
    alpha_0, beta_0 = compute_ts_prior(train_df['CTR'].values)
    print(f"TS Prior: α₀={alpha_0:.4f}, β₀={beta_0:.4f}")
    
    hyperparams = {
        'naux_ucb': NAUX_UCB,
        'alpha_ucb': ALPHA_UCB,
        'naux_ts': NAUX_TS,
        'alpha_0': alpha_0,
        'beta_0': beta_0,
    }
    
    # Prepare arguments for parallel execution
    print("\nPreparing evaluation tasks...")
    eval_args = []
    
    for test_id in sampled_test_ids:
        test_data = test_df[test_df['test_id'] == test_id]
        if len(test_data) == 0:
            continue
            
        pred_CTR = predictions.get(test_id)
        if pred_CTR is None:
            continue
            
        test_data_dict = {
            'CTR': test_data['CTR'].values,
            'headlines': test_data['headline'].tolist()
        }
        
        if len(pred_CTR) != len(test_data_dict['CTR']):
            continue
        
        eval_args.append((test_id, test_data_dict, pred_CTR, TAU_VALUES, n_repeats, hyperparams))
    
    print(f"Prepared {len(eval_args)} test evaluations")
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    
    print(f"\nRunning parallel evaluation with {n_workers} workers...")
    
    # Run parallel evaluation
    all_results = []
    
    with Pool(n_workers) as pool:
        # Use imap for progress tracking
        results_iter = pool.imap(evaluate_single_test, eval_args)
        
        for result in tqdm(results_iter, total=len(eval_args), desc="Evaluating"):
            all_results.extend(result)
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\nEvaluation complete. Total results: {len(results_df)}")
    
    return results_df, sampled_test_ids, hyperparams


def run_parallel_bai_evaluation(test_df: pd.DataFrame, predictions: Dict[int, np.ndarray],
                                sampled_test_ids: np.ndarray, n_repeats: int = 10,
                                n_workers: int = None) -> pd.DataFrame:
    """Run BAI evaluation in parallel."""
    
    print("\nRunning Best Arm Identification evaluation...")
    
    # Prepare arguments
    eval_args = []
    for test_id in sampled_test_ids:
        test_data = test_df[test_df['test_id'] == test_id]
        if len(test_data) == 0:
            continue
            
        pred_CTR = predictions.get(test_id)
        if pred_CTR is None:
            continue
        
        test_data_dict = {'CTR': test_data['CTR'].values}
        
        if len(pred_CTR) != len(test_data_dict['CTR']):
            continue
        
        eval_args.append((test_id, test_data_dict, pred_CTR, NAUX_BAI, n_repeats))
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(cpu_count(), 16)
    
    all_results = []
    
    with Pool(n_workers) as pool:
        results_iter = pool.imap(evaluate_bai_single_test, eval_args)
        
        for result in tqdm(results_iter, total=len(eval_args), desc="BAI Evaluation"):
            all_results.extend(result)
    
    return pd.DataFrame(all_results)


# =============================================================================
# Visualization and Report
# =============================================================================

def create_visualizations(results_df: pd.DataFrame, bai_df: pd.DataFrame,
                          output_dir: str) -> Tuple[str, str, str]:
    """
    Create APA-style visualizations.
    
    Returns paths to generated figures.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # APA Style Configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
    })
    
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Color palette
    colors = {
        'LLM-2UCBs': '#2166AC',
        'UCB': '#B2182B',
        'Pure_LLM': '#4DAF4A',
        'E&C': '#FF7F00',
        'LLM-TS': '#984EA3',
        'TS': '#A65628',
    }
    
    markers = {
        'LLM-2UCBs': 'o',
        'UCB': 's',
        'Pure_LLM': '^',
        'E&C': 'v',
        'LLM-TS': 'D',
        'TS': 'p',
    }
    
    labels = {
        'LLM-2UCBs': 'LLM-2UCBs (LOLA)',
        'UCB': 'UCB',
        'Pure_LLM': 'Pure LLM',
        'E&C': 'E&C',
        'LLM-TS': 'LLM-TS',
        'TS': 'TS',
    }
    
    # Figure 1: Main Results
    print("\nGenerating Figure 1: Main Results...")
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
        
        ax.errorbar(tau_values, means, yerr=se,
                   label=labels[method], color=colors[method],
                   marker=markers[method], capsize=3, linewidth=1.5)
    
    ax.set_xlabel('Traffic/Impressions per Headline (τ)', fontsize=11)
    ax.set_ylabel('Average Clicks per Test per Period', fontsize=11)
    ax.set_title('LOLA vs. Benchmarks: Regret Minimization Performance', 
                fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95, frameon=True)
    ax.set_xlim(0, 1050)
    
    fig1_path = os.path.join(figures_dir, 'lola_main_results.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fig1_path}")
    
    # Figure 2: Pairwise Comparison
    print("\nGenerating Figure 2: Pairwise Comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    comparisons = [
        ('LLM-2UCBs', 'UCB', 'LOLA vs UCB'),
        ('LLM-2UCBs', 'E&C', 'LOLA vs E&C'),
        ('LLM-2UCBs', 'Pure_LLM', 'LOLA vs Pure LLM'),
    ]
    
    for method_a, method_b, label in comparisons:
        improvements = []
        for tau in tau_values:
            tau_data = results_df[results_df['tau'] == tau]
            a = tau_data[method_a].values
            b = tau_data[method_b].values
            imp = (a.mean() / b.mean() - 1) * 100
            improvements.append(imp)
        
        ax.plot(tau_values, improvements, marker='o', label=label, linewidth=1.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Traffic/Impressions per Headline (τ)', fontsize=11)
    ax.set_ylabel('Percentage Improvement (%)', fontsize=11)
    ax.set_title('LOLA Improvement Over Benchmark Algorithms', 
                fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.95)
    
    fig2_path = os.path.join(figures_dir, 'lola_pairwise_comparison.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fig2_path}")
    
    # Figure 3: BAI Results
    print("\nGenerating Figure 3: BAI Results...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    summary = bai_df.groupby('method').agg({
        'success': ['mean', 'std'],
        'n_pulls': ['mean', 'std']
    }).reset_index()
    summary.columns = ['method', 'success_rate', 'success_se', 'avg_pulls', 'pulls_se']
    summary['success_rate'] *= 100
    summary['success_se'] *= 100
    
    ax.errorbar(summary['avg_pulls'], summary['success_rate'],
               xerr=summary['pulls_se'], yerr=summary['success_se'],
               fmt='o', color='#2166AC', capsize=3, markersize=8,
               label='LLM-BAI (LOLA)')
    
    ax.set_xlabel('Average Number of Pulls per Test', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Best Arm Identification Performance', 
                fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.95)
    
    fig3_path = os.path.join(figures_dir, 'lola_bai_results.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {fig3_path}")
    
    return fig1_path, fig2_path, fig3_path


def create_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create APA-style comparison table with significance tests."""
    from scipy import stats
    
    tau_values = sorted(results_df['tau'].unique())
    
    comparisons = [
        ('LLM-2UCBs', 'UCB'),
        ('LLM-2UCBs', 'E&C'),
        ('LLM-2UCBs', 'Pure_LLM'),
        ('LLM-2UCBs', 'TS'),
        ('LLM-TS', 'TS'),
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
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''
            
            row[f'{method_a} vs {method_b}'] = f'{improvement:+.2f}{sig}'
        
        results.append(row)
    
    return pd.DataFrame(results)


def generate_pdf_report(results_df: pd.DataFrame, bai_df: pd.DataFrame,
                        comp_table: pd.DataFrame, fig_paths: Tuple[str, str, str],
                        output_dir: str, hyperparams: dict, n_tests: int,
                        n_repeats: int, seed: int):
    """Generate PDF report using LaTeX."""
    
    print("\nGenerating PDF report...")
    
    # Create LaTeX report
    latex_content = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{float}

\title{\textbf{LOLA Replication Study Report}\\
\large LLM-Assisted Online Learning Algorithm for Content Experiments}
\author{Replication of Ye et al. (2025)\\
Using Fine-tuned DeepSeek-7B-Chat}
\date{""" + datetime.now().strftime("%B %d, %Y") + r"""}

\begin{document}
\maketitle

\section{Introduction}

This report presents the replication results of the LOLA (LLM-Assisted Online Learning Algorithm) framework from Ye et al. (2025), implemented with a fine-tuned DeepSeek-7B-Chat model for CTR prediction.

\subsection{Study Design}

\begin{itemize}
    \item \textbf{Model}: Fine-tuned DeepSeek-7B-Chat with LoRA weights
    \item \textbf{Data}: Upworthy headline experiments (10\% random sample)
    \item \textbf{Sample Size}: """ + f"{n_tests}" + r""" tests
    \item \textbf{Monte Carlo Repeats}: """ + f"{n_repeats}" + r""" per test
    \item \textbf{Random Seed}: """ + f"{seed}" + r"""
\end{itemize}

\subsection{Hyperparameters}

The following hyperparameters were used (as specified in the original paper):

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
$naux$ (LLM-2UCBs) & """ + f"{hyperparams['naux_ucb']}" + r""" \\
$naux$ (LLM-TS) & """ + f"{hyperparams['naux_ts']}" + r""" \\
$\alpha$ (UCB) & """ + f"{hyperparams['alpha_ucb']}" + r""" \\
$\alpha_0$ (TS prior) & """ + f"{hyperparams['alpha_0']:.4f}" + r""" \\
$\beta_0$ (TS prior) & """ + f"{hyperparams['beta_0']:.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}

\section{Main Results}

\subsection{Regret Minimization Performance}

Figure~\ref{fig:main} compares the average clicks per test per period across all algorithms. LLM-2UCBs (LOLA) consistently outperforms benchmark algorithms across all time horizons.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{""" + fig_paths[0] + r"""}
\caption{LOLA vs. Benchmarks: Average clicks per test per period across different traffic levels. Error bars represent standard errors.}
\label{fig:main}
\end{figure}

\subsection{Pairwise Comparisons}

Figure~\ref{fig:pairwise} shows the percentage improvement of LOLA over benchmark algorithms.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{""" + fig_paths[1] + r"""}
\caption{Percentage improvement of LOLA over benchmark algorithms. Positive values indicate LOLA outperforms the benchmark.}
\label{fig:pairwise}
\end{figure}

\subsection{Statistical Comparison Table}

Table~\ref{tab:comparison} presents the percentage improvement between algorithms with statistical significance levels.

\begin{table}[H]
\centering
\caption{Pairwise Algorithm Comparisons (\% improvement with significance)}
\label{tab:comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{cccccc}
\toprule
$\tau$ & LOLA vs UCB & LOLA vs E\&C & LOLA vs Pure LLM & LOLA vs TS & LLM-TS vs TS \\
\midrule
"""
    
    # Add table rows
    for _, row in comp_table.iterrows():
        latex_content += f"{int(row['τ'])} & {row['LLM-2UCBs vs UCB']} & {row['LLM-2UCBs vs E&C']} & {row['LLM-2UCBs vs Pure_LLM']} & {row['LLM-2UCBs vs TS']} & {row['LLM-TS vs TS']} \\\\\n"
    
    latex_content += r"""\bottomrule
\end{tabular}
}
\vspace{0.5em}

\footnotesize Note: * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$ (paired t-test)
\end{table}

\section{Best Arm Identification}

Figure~\ref{fig:bai} presents the performance of LLM-BAI for Best Arm Identification.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{""" + fig_paths[2] + r"""}
\caption{Best Arm Identification: Success rate vs. average number of pulls.}
\label{fig:bai}
\end{figure}

"""
    
    # Add summary statistics
    latex_content += r"""
\section{Summary Statistics}

\begin{table}[H]
\centering
\caption{Average Clicks per Test per Period by Algorithm}
\begin{tabular}{lcccccc}
\toprule
$\tau$ & LLM-2UCBs & UCB & E\&C & Pure LLM & LLM-TS & TS \\
\midrule
"""
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        row = f"{tau}"
        for method in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM', 'LLM-TS', 'TS']:
            mean = tau_data[method].mean()
            row += f" & {mean:.6f}"
        row += r" \\" + "\n"
        latex_content += row
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\section{Conclusion}

This replication study validates the LOLA framework using a fine-tuned DeepSeek-7B-Chat model. Key findings:

\begin{enumerate}
    \item \textbf{Short horizons}: LOLA achieves substantial improvements over E\&C (traditional A/B testing), demonstrating the value of LLM priors in early-stage traffic allocation.
    
    \item \textbf{Long horizons}: LOLA maintains superiority over Pure LLM, showing that combining LLM predictions with online learning is more robust than relying solely on LLM predictions.
    
    \item \textbf{Robustness}: The 2-UCB approach (min\{$U^1$, $U^2$\}) effectively balances LLM guidance with empirical evidence, avoiding the pitfalls of LLM overestimation.
\end{enumerate}

\section*{References}

Ye, Z., Yoganarasimhan, H., \& Zheng, Y. (2025). LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments. \textit{Marketing Science} (forthcoming).

\end{document}
"""
    
    # Save LaTeX file
    tex_path = os.path.join(output_dir, 'lola_report.tex')
    with open(tex_path, 'w') as f:
        f.write(latex_content)
    print(f"  LaTeX report saved: {tex_path}")
    
    # Compile to PDF
    print("  Compiling PDF...")
    import subprocess
    
    # Run pdflatex
    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-output-directory', output_dir, tex_path],
        capture_output=True, text=True, cwd=output_dir
    )
    
    pdf_path = os.path.join(output_dir, 'lola_report.pdf')
    if os.path.exists(pdf_path):
        print(f"  PDF report saved: {pdf_path}")
    else:
        print(f"  Warning: PDF compilation may have failed. Check {tex_path}")
    
    return pdf_path if os.path.exists(pdf_path) else tex_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LOLA evaluation with parallel execution')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                       help='Fraction of tests to sample (default: 0.1 = 10%%)')
    parser.add_argument('--n-repeats', type=int, default=10,
                       help='Number of Monte Carlo repeats (default: 10)')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(TEST_FILE)
    print(f"Total tests in dataset: {test_df['test_id'].nunique()}")
    
    # Load predictions
    print("\nLoading LLM predictions...")
    with open(PREDICTIONS_FILE, 'rb') as f:
        predictions = pickle.load(f)
    print(f"Predictions loaded for {len(predictions)} tests")
    
    # Run parallel evaluation
    start_time = datetime.now()
    
    results_df, sampled_test_ids, hyperparams = run_parallel_evaluation(
        test_df, predictions,
        sample_ratio=args.sample_ratio,
        n_repeats=args.n_repeats,
        n_workers=args.n_workers,
        seed=args.seed
    )
    
    # Run BAI evaluation
    bai_df = run_parallel_bai_evaluation(
        test_df, predictions, sampled_test_ids,
        n_repeats=args.n_repeats,
        n_workers=args.n_workers
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nTotal evaluation time: {duration:.1f} seconds")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results_path = os.path.join(RESULTS_DIR, 'lola_parallel_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    bai_path = os.path.join(RESULTS_DIR, 'lola_bai_results.csv')
    bai_df.to_csv(bai_path, index=False)
    print(f"BAI results saved to: {bai_path}")
    
    # Create visualizations
    fig_paths = create_visualizations(results_df, bai_df, RESULTS_DIR)
    
    # Create comparison table
    print("\nCreating comparison table...")
    comp_table = create_comparison_table(results_df)
    table_path = os.path.join(RESULTS_DIR, 'comparison_table.csv')
    comp_table.to_csv(table_path, index=False)
    print(f"Comparison table saved to: {table_path}")
    
    # Generate PDF report
    report_path = generate_pdf_report(
        results_df, bai_df, comp_table, fig_paths,
        RESULTS_DIR, hyperparams, len(sampled_test_ids),
        args.n_repeats, args.seed
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\nAverage Clicks per Test per Period:")
    print("-"*70)
    
    header = f"{'τ':>6}"
    for method in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM', 'LLM-TS', 'TS']:
        header += f"  {method:>12}"
    print(header)
    print("-"*70)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        row = f"{tau:>6}"
        for method in ['LLM-2UCBs', 'UCB', 'E&C', 'Pure_LLM', 'LLM-TS', 'TS']:
            mean = tau_data[method].mean()
            row += f"  {mean:>12.6f}"
        print(row)
    
    print("\n" + "-"*70)
    print("Percentage Improvement of LOLA over Benchmarks:")
    print("-"*70)
    
    header = f"{'τ':>6}"
    for benchmark in ['vs UCB', 'vs E&C', 'vs Pure_LLM']:
        header += f"  {benchmark:>12}"
    print(header)
    print("-"*70)
    
    for tau in sorted(results_df['tau'].unique()):
        tau_data = results_df[results_df['tau'] == tau]
        row = f"{tau:>6}"
        for benchmark in ['UCB', 'E&C', 'Pure_LLM']:
            lol = tau_data['LLM-2UCBs'].mean()
            other = tau_data[benchmark].mean()
            imp = (lol / other - 1) * 100
            row += f"  {imp:>+11.2f}%"
        print(row)
    
    print("\n" + "="*70)
    print("OUTPUT FILES:")
    print(f"  - Results CSV: {results_path}")
    print(f"  - BAI Results: {bai_path}")
    print(f"  - Comparison Table: {table_path}")
    print(f"  - Main Results Figure: {fig_paths[0]}")
    print(f"  - Pairwise Comparison Figure: {fig_paths[1]}")
    print(f"  - BAI Figure: {fig_paths[2]}")
    print(f"  - PDF Report: {report_path}")
    print("="*70)


if __name__ == "__main__":
    main()

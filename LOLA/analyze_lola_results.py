"""
LOLA Results Analysis and Visualization Script
Uses existing simulation results to generate plots and comparison tables
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

print('='*80)
print('LOLA RESULTS ANALYSIS')
print('='*80)

# Load existing results
results_path = r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\LLM_News\LOLA - Regret Minimize\simulation_results_regret_min.csv'
results_df = pd.read_csv(results_path)
print(f'Loaded existing results from {results_path}')
print(f'T values: {results_df["T"].tolist()}')

# Create output directory
output_path = Path(r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\results')
output_path.mkdir(exist_ok=True)

# Helper function to safely parse list strings
def parse_list(s):
    try:
        return eval(s)
    except:
        return np.array([])

# ============================================================================
# PLOT 1: Main Comparison (LOLA vs UCB vs Pure LLM vs E&C vs UCB+LLM)
# ============================================================================
print('\nGenerating Plot 1: Main Algorithm Comparison...')

# Reload fresh data for plotting
results_df = pd.read_csv(results_path)
plt.figure(figsize=(10, 6))
results_df['T'] = pd.to_numeric(results_df['T'], errors='coerce')

algorithms = [
    ('ucb2_result', 'LOLA (LLM-2UCBs)', 'o', '-', 'blue'),
    ('one_ucb_result', 'UCB with LLM priors', 'o', '-', 'black'),
    ('ucb1_result', 'UCB', 's', '--', 'green'),
    ('no_bandit_result', 'Pure LLM', 'D', '-.', 'red'),
    ('ab_result', 'E&C', '^', ':', 'purple')
]

for col, label, marker, linestyle, color in algorithms:
    results_df[col] = results_df[col].apply(parse_list)
    results_df[f'{label}_mean'] = results_df.apply(
        lambda row: np.mean(np.array([row[col]]) / row['T']) if len(row[col]) > 0 else np.nan, axis=1)
    plt.plot(results_df['T'], results_df[f'{label}_mean'],
            marker=marker, linestyle=linestyle, color=color, label=label)

plt.xlabel(r'Traffic/Impressions per headline in tests $\tau$')
plt.ylabel('Average clicks per test per period')
plt.xticks([50, 100, 200, 400, 600, 800, 1000])
plt.xlim([50, 1000])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot1_path = output_path / 'lola_comparison.pdf'
plt.savefig(plot1_path, format='pdf', bbox_inches='tight')
print(f'  Saved: {plot1_path}')
plt.close()

# ============================================================================
# SUMMARY TABLE - Direct computation from parsed data
# ============================================================================
print('\n' + '='*80)
print('SUMMARY TABLE: Average Clicks Per Period')
print('='*80)
print(f"{'T':<8} {'LOLA':<10} {'UCB':<10} {'UCB+LLM':<10} {'Pure LLM':<10} {'E&C':<10}")
print('-'*80)

summary_data = []
for _, row in results_df.iterrows():
    T = row['T']
    lola_arr = np.array(row['ucb2_result'])
    ucb_arr = np.array(row['ucb1_result'])
    one_ucb_arr = np.array(row['one_ucb_result'])
    no_bandit_arr = np.array(row['no_bandit_result'])
    ab_arr = np.array(row['ab_result'])

    lola_mean = np.mean(lola_arr) / T if len(lola_arr) > 0 else np.nan
    ucb_mean = np.mean(ucb_arr) / T if len(ucb_arr) > 0 else np.nan
    one_ucb_mean = np.mean(one_ucb_arr) / T if len(one_ucb_arr) > 0 else np.nan
    no_bandit_mean = np.mean(no_bandit_arr) / T if len(no_bandit_arr) > 0 else np.nan
    ab_mean = np.mean(ab_arr) / T if len(ab_arr) > 0 else np.nan

    print(f"{T:<8} {lola_mean:<10.6f} {ucb_mean:<10.6f} {one_ucb_mean:<10.6f} "
          f"{no_bandit_mean:<10.6f} {ab_mean:<10.6f}")

    summary_data.append({
        'T': T,
        'LOLA': lola_mean,
        'UCB': ucb_mean,
        'UCB+LLM': one_ucb_mean,
        'Pure LLM': no_bandit_mean,
        'E&C': ab_mean,
        'LOLA_vs_UCB_%': ((lola_mean - ucb_mean) / ucb_mean * 100) if ucb_mean > 0 else np.nan,
        'LOLA_vs_PureLLM_%': ((lola_mean - no_bandit_mean) / no_bandit_mean * 100) if no_bandit_mean > 0 else np.nan,
        'LOLA_vs_EC_%': ((lola_mean - ab_mean) / ab_mean * 100) if ab_mean > 0 else np.nan
    })

print('='*80)

# ============================================================================
# KEY IMPROVEMENTS TABLE
# ============================================================================
print('\nKEY IMPROVEMENTS (LOLA vs Baselines)')
print('='*80)
print(f"{'T':<8} {'vs UCB':<12} {'vs Pure LLM':<15} {'vs E&C':<12}")
print('-'*80)

for item in summary_data:
    T = item['T']
    vs_ucb = item['LOLA_vs_UCB_%']
    vs_llm = item['LOLA_vs_PureLLM_%']
    vs_ec = item['LOLA_vs_EC_%']

    print(f"{T:<8} {vs_ucb:+.2f}%       {vs_llm:+.2f}%         {vs_ec:+.2f}%")

print('='*80)

# Save summary tables
summary_df = pd.DataFrame(summary_data)
summary_path = output_path / 'summary_table.csv'
summary_df.to_csv(summary_path, index=False)
print(f'\nSummary table saved to: {summary_path}')

# ============================================================================
# STATISTICAL COMPARISON TABLE
# ============================================================================
print('\nGenerating Statistical Comparison Table...')

label_to_column = {
    'LOLA': 'ucb2_result',
    'UCB': 'ucb1_result',
    'Pure LLM': 'no_bandit_result',
    'E&C': 'ab_result',
    'UCB+LLM': 'one_ucb_result'
}

import itertools
comparisons = list(itertools.combinations(label_to_column.keys(), 2))

comparison_df = pd.DataFrame({'T': results_df['T']})

for label1, label2 in comparisons:
    col1 = label_to_column[label1]
    col2 = label_to_column[label2]

    mean_diffs = []
    p_values = []

    for _, row in results_df.iterrows():
        arr1 = np.array(row[col1])
        arr2 = np.array(row[col2])
        data1 = arr1 / row['T']
        data2 = arr2 / row['T']

        # Skip if empty arrays
        if len(data1) == 0 or len(data2) == 0:
            mean_diffs.append(np.nan)
            p_values.append(np.nan)
            continue

        mean_diff = np.mean(data1 - data2) / np.mean(data2) * 100
        t_stat, p_value = ttest_rel(data1, data2)

        mean_diffs.append(mean_diff)
        p_values.append(p_value)

    comparison_df[f'{label1} vs {label2}_diff_%'] = mean_diffs
    comparison_df[f'{label1} vs {label2}_pval'] = p_values

table_path = output_path / 'statistical_comparison.csv'
comparison_df.to_csv(table_path, index=False)
print(f'  Saved: {table_path}')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
print(f'\nGenerated files:')
print(f'  1. {plot1_path}')
print(f'  2. {table_path}')
print(f'  3. {summary_path}')

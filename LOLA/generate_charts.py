"""
LOLA Replication - Comprehensive Report Generator
Generates charts and English report for the LOLA algorithm replication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set up matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

print('='*80)
print('LOLA REPLICATION - Report Generator')
print('='*80)

# Paths
base_path = Path(r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication')
results_path = base_path / 'LLM_News' / 'LOLA - Regret Minimize' / 'simulation_results_regret_min.csv'
output_path = base_path / 'results'
output_path.mkdir(exist_ok=True)

# Load results
results_df = pd.read_csv(results_path)

# Helper function to parse list strings
def parse_list(s):
    try:
        return eval(s)
    except:
        return np.array([])

# Parse all result columns
result_columns = ['ucb2_result', 'ucb1_result', 'one_ucb_result', 'no_bandit_result', 'ab_result']
for col in result_columns:
    results_df[col] = results_df[col].apply(parse_list)

# ============================================================================
# CHART 1: Main Comparison Plot (Enhanced)
# ============================================================================
print('\nGenerating Chart 1: Main Algorithm Comparison...')

fig, ax = plt.subplots(figsize=(12, 7))

algorithms = [
    ('ucb2_result', 'LOLA (LLM-2UCBs)', 'o', '-', '#2563eb', 8),
    ('one_ucb_result', 'UCB with LLM priors', 's', '-', '#475569', 7),
    ('ucb1_result', 'UCB', 's', '--', '#16a34a', 7),
    ('no_bandit_result', 'Pure LLM', '^', '-.', '#dc2626', 7),
    ('ab_result', 'E&C', 'd', ':', '#7c3aed', 7)
]

for col, label, marker, linestyle, color, size in algorithms:
    means = []
    for _, row in results_df.iterrows():
        arr = np.array(row[col])
        if len(arr) > 0:
            means.append(np.mean(arr) / row['T'])
        else:
            means.append(np.nan)
    ax.plot(results_df['T'], means, marker=marker, markersize=size,
            linestyle=linestyle, linewidth=2, color=color, label=label)

ax.set_xlabel(r'Traffic per Headline $\tau$', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Clicks per Period', fontsize=12, fontweight='bold')
ax.set_xticks([50, 100, 200, 400, 600, 800, 1000])
ax.set_xlim([45, 1050])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.set_title('LOLA vs Baseline Algorithms: Performance Comparison',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(output_path / 'chart1_main_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig(output_path / 'chart1_main_comparison.png', format='png', bbox_inches='tight', dpi=300)
print(f'  Saved: chart1_main_comparison.pdf/png')
plt.close()

# ============================================================================
# CHART 2: Performance Improvement Over UCB
# ============================================================================
print('\nGenerating Chart 2: LOLA Improvement Over UCB...')

fig, ax = plt.subplots(figsize=(10, 6))

T_values = results_df['T'].values
improvements = []

for _, row in results_df.iterrows():
    T = row['T']
    lola_mean = np.mean(np.array(row['ucb2_result'])) / T
    ucb_mean = np.mean(np.array(row['ucb1_result'])) / T
    improvement = (lola_mean - ucb_mean) / ucb_mean * 100
    improvements.append(improvement)

bars = ax.bar(range(len(T_values)), improvements, color='#2563eb', alpha=0.7, edgecolor='white')
ax.set_xticks(range(len(T_values)))
ax.set_xticklabels(T_values)
ax.set_xlabel('Traffic per Headline (T)', fontsize=11, fontweight='bold')
ax.set_ylabel('Improvement over UCB (%)', fontsize=11, fontweight='bold')
ax.set_title('LOLA Performance Gain vs Standard UCB', fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'+{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_path / 'chart2_improvement_over_ucb.pdf', format='pdf', bbox_inches='tight')
plt.savefig(output_path / 'chart2_improvement_over_ucb.png', format='png', bbox_inches='tight', dpi=300)
print(f'  Saved: chart2_improvement_over_ucb.pdf/png')
plt.close()

# ============================================================================
# CHART 3: All Algorithms Performance Heatmap
# ============================================================================
print('\nGenerating Chart 3: Performance Heatmap...')

fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for heatmap
heatmap_data = []
algorithm_names = ['LOLA', 'UCB', 'UCB+LLM', 'Pure LLM', 'E&C']
for _, row in results_df.iterrows():
    T = row['T']
    row_data = [T]
    row_data.extend([np.mean(np.array(row[col])) / T for col in
                     ['ucb2_result', 'ucb1_result', 'one_ucb_result', 'no_bandit_result', 'ab_result']])
    heatmap_data.append(row_data)

heatmap_df = pd.DataFrame(heatmap_data, columns=['T'] + algorithm_names)
heatmap_df = heatmap_df.set_index('T')

im = ax.imshow(heatmap_df.T.values, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(heatmap_df.index)))
ax.set_yticks(range(len(algorithm_names)))
ax.set_xticklabels(heatmap_df.index)
ax.set_yticklabels(algorithm_names)
ax.set_xlabel('Traffic per Headline (T)', fontsize=11, fontweight='bold')
ax.set_ylabel('Algorithm', fontsize=11, fontweight='bold')
ax.set_title('Algorithm Performance Heatmap\n(Average Clicks per Period)',
             fontsize=13, fontweight='bold', pad=15)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Average Clicks per Period', rotation=270, labelpad=15)

# Add value annotations
for i in range(len(heatmap_df.index)):
    for j in range(len(algorithm_names)):
        val = heatmap_df.T.iloc[j, i]
        text = ax.text(i, j, f'{val:.4f}', ha='center', va='center',
                      color='black' if val < 0.052 else 'white', fontsize=8)

plt.tight_layout()
plt.savefig(output_path / 'chart3_performance_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.savefig(output_path / 'chart3_performance_heatmap.png', format='png', bbox_inches='tight', dpi=300)
print(f'  Saved: chart3_performance_heatmap.pdf/png')
plt.close()

# ============================================================================
# CHART 4: Relative Performance Bar Chart
# ============================================================================
print('\nGenerating Chart 4: Relative Performance Comparison...')

fig, ax = plt.subplots(figsize=(12, 6))

T_values = results_df['T'].values
x = np.arange(len(T_values))
width = 0.15

algorithms_bar = [
    ('ucb2_result', 'LOLA', '#2563eb'),
    ('ucb1_result', 'UCB', '#16a34a'),
    ('one_ucb_result', 'UCB+LLM', '#475569'),
    ('no_bandit_result', 'Pure LLM', '#dc2626'),
    ('ab_result', 'E&C', '#7c3aed')
]

for i, (col, name, color) in enumerate(algorithms_bar):
    values = []
    for _, row in results_df.iterrows():
        arr = np.array(row[col])
        if len(arr) > 0:
            values.append(np.mean(arr) / row['T'])
        else:
            values.append(0)
    ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.8)

ax.set_xlabel('Traffic per Headline (T)', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Clicks per Period', fontsize=11, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(T_values)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.set_title('Algorithm Performance Across Different Traffic Levels',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig(output_path / 'chart4_relative_performance.pdf', format='pdf', bbox_inches='tight')
plt.savefig(output_path / 'chart4_relative_performance.png', format='png', bbox_inches='tight', dpi=300)
print(f'  Saved: chart4_relative_performance.pdf/png')
plt.close()

print('\n' + '='*80)
print('ALL CHARTS GENERATED SUCCESSFULLY')
print('='*80)

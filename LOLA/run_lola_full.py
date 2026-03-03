"""
LOLA Full Replication Script (Simplified - No Thompson Sampling)
Runs 5 core benchmark algorithms across all T values

Algorithms:
1. LOLA (LLM-2UCBs) - Proposed Method
2. Standard UCB
3. UCB with LLM Priors
4. Pure LLM (No Bandit)
5. Explore-then-Commit (E&C)
"""

import pandas as pd
import numpy as np
import ast
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print('='*60)
print('LOLA REPLICATION - Full Simulation (5 Algorithms)')
print('='*60)

# Load data
data_path = r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\LLM_News\LOLA - Regret Minimize\LoRA CTR.csv'
test_df = pd.read_csv(data_path)
n_tests = test_df.groupby('test_id').ngroups
print(f'Loaded {len(test_df)} headlines from {n_tests} tests')

# Parameters
fixed_n_llm = 1000
best_z_ucb = 0.08
best_ratio = 0.2

T_values = [50, 100, 200, 400, 600, 800, 1000]
all_results = []

# Progress tracking
total_tasks = len(T_values) * 5
current_task = 0

for T in T_values:
    print(f'\n{"="*50}')
    print(f'T = {T} (Traffic per headline)')
    print(f'{"="*50}')

    result_row = {'T': T}

    # Algorithm 1: Standard UCB (no LLM prior)
    current_task += 1
    print(f'  [{current_task}/{total_tasks}] Standard UCB...')
    sys.stdout.flush()
    rng = np.random.RandomState(42)
    ucb1_results = []
    for test_id, group in test_df.groupby('test_id'):
        n_arms = len(group)
        trials = np.ones(n_arms)
        successes = np.zeros(n_arms)
        total_T = int(n_arms * T)
        rewards = np.zeros(total_T)
        for t in range(total_T):
            mean_val = successes / trials
            ucb = mean_val + best_z_ucb * np.sqrt(np.log(t + 2) / trials)
            chosen_arm = np.argmax(ucb)
            ctr = group.iloc[chosen_arm]['CTR']
            reward = rng.binomial(1, ctr)
            trials[chosen_arm] += 1
            successes[chosen_arm] += reward
            rewards[t] = reward
        ucb1_results.append(np.sum(rewards))
    result_row['ucb1_result'] = ucb1_results

    # Algorithm 2: LOLA (LLM-2UCBs) - min(standard UCB, LLM UCB)
    current_task += 1
    print(f'  [{current_task}/{total_tasks}] LOLA (LLM-2UCBs)...')
    sys.stdout.flush()
    rng = np.random.RandomState(42)
    ucb2_results = []
    for test_id, group in test_df.groupby('test_id'):
        n_arms = len(group)
        trials = np.ones(n_arms)
        successes = np.zeros(n_arms)
        ctr_llm = np.array(group['ini_CTR'].values)
        n_llm = fixed_n_llm * np.ones(n_arms)
        total_T = int(n_arms * T)
        rewards = np.zeros(total_T)
        for t in range(total_T):
            mean_val = successes / trials
            ucb = mean_val + best_z_ucb * np.sqrt(np.log(t + 2) / trials)
            ucb_llm = (ctr_llm * n_llm + successes) / (trials + n_llm) + best_z_ucb * np.sqrt(np.log(t + 2) / (trials + n_llm))
            ucbs = np.minimum(ucb, ucb_llm)  # LOLA: take minimum
            chosen_arm = np.argmax(ucbs)
            ctr = group.iloc[chosen_arm]['CTR']
            reward = rng.binomial(1, ctr)
            trials[chosen_arm] += 1
            successes[chosen_arm] += reward
            rewards[t] = reward
        ucb2_results.append(np.sum(rewards))
    result_row['ucb2_result'] = ucb2_results

    # Algorithm 3: UCB with LLM priors only (not taking minimum)
    current_task += 1
    print(f'  [{current_task}/{total_tasks}] UCB with LLM priors...')
    sys.stdout.flush()
    rng = np.random.RandomState(42)
    one_ucb_results = []
    for test_id, group in test_df.groupby('test_id'):
        n_arms = len(group)
        trials = np.ones(n_arms)
        successes = np.zeros(n_arms)
        ctr_llm = np.array(group['ini_CTR'].values)
        n_llm = fixed_n_llm * np.ones(n_arms)
        total_T = int(n_arms * T)
        rewards = np.zeros(total_T)
        for t in range(total_T):
            mean_val = successes / trials
            ucb_llm = (ctr_llm * n_llm + successes) / (trials + n_llm) + best_z_ucb * np.sqrt(np.log(t + 2) / (trials + n_llm))
            chosen_arm = np.argmax(ucb_llm)
            ctr = group.iloc[chosen_arm]['CTR']
            reward = rng.binomial(1, ctr)
            trials[chosen_arm] += 1
            successes[chosen_arm] += reward
            rewards[t] = reward
        one_ucb_results.append(np.sum(rewards))
    result_row['one_ucb_result'] = one_ucb_results

    # Algorithm 4: Pure LLM (no bandit, always select highest predicted CTR)
    current_task += 1
    print(f'  [{current_task}/{total_tasks}] Pure LLM...')
    sys.stdout.flush()
    rng = np.random.RandomState(42)
    no_bandit_results = []
    for test_id, group in test_df.groupby('test_id'):
        n_arms = len(group)
        total_T = int(n_arms * T)
        best_arm = np.argmax(group['ini_CTR'])
        ctr = group.iloc[best_arm]['CTR']
        rewards = rng.binomial(1, ctr, total_T)
        no_bandit_results.append(np.sum(rewards))
    result_row['no_bandit_result'] = no_bandit_results

    # Algorithm 5: Explore-then-Commit (20% exploration, 80% exploitation)
    current_task += 1
    print(f'  [{current_task}/{total_tasks}] Explore-then-Commit...')
    sys.stdout.flush()
    rng = np.random.RandomState(42)
    ab_results = []
    for test_id, group in test_df.groupby('test_id'):
        n_arms = len(group)
        total_T = int(n_arms * T)
        trials = np.ones(n_arms)
        successes = np.zeros(n_arms)
        explore_T = int(best_ratio * total_T)
        exploit_T = total_T - explore_T
        rewards = np.zeros(total_T)
        chosen_arms = rng.choice(n_arms, explore_T)
        for i, arm in enumerate(chosen_arms):
            ctr = group.iloc[arm]['CTR']
            reward = rng.binomial(1, ctr)
            trials[arm] += 1
            successes[arm] += reward
            rewards[i] = reward
        empirical_ctrs = successes / trials
        best_arm = np.argmax(empirical_ctrs)
        ctr = group.iloc[best_arm]['CTR']
        rewards[explore_T:total_T] = rng.binomial(1, ctr, exploit_T)
        ab_results.append(np.sum(rewards))
    result_row['ab_result'] = ab_results

    # Print summary for this T
    print(f'\n  Results for T={T}:')
    print(f'    LOLA:      {np.mean(ucb2_results)/T:.6f}')
    print(f'    UCB:       {np.mean(ucb1_results)/T:.6f}')
    print(f'    UCB+LLM:   {np.mean(one_ucb_results)/T:.6f}')
    print(f'    Pure LLM:  {np.mean(no_bandit_results)/T:.6f}')
    print(f'    E&C:       {np.mean(ab_results)/T:.6f}')

    # Calculate improvements
    lola_vs_ucb = (np.mean(ucb2_results) - np.mean(ucb1_results)) / np.mean(ucb1_results) * 100
    lola_vs_llm = (np.mean(ucb2_results) - np.mean(no_bandit_results)) / np.mean(no_bandit_results) * 100
    lola_vs_ec = (np.mean(ucb2_results) - np.mean(ab_results)) / np.mean(ab_results) * 100

    print(f'\n  LOLA improvements:')
    print(f'    vs UCB:      {lola_vs_ucb:+.2f}%')
    print(f'    vs Pure LLM: {lola_vs_llm:+.2f}%')
    print(f'    vs E&C:      {lola_vs_ec:+.2f}%')

    all_results.append(result_row)

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Save results
output_path = Path(r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\results')
output_path.mkdir(exist_ok=True)
results_file = output_path / 'simulation_results.csv'
results_df.to_csv(results_file, index=False)
print(f'\nResults saved to {results_file}')

# Print final summary table
print('\n' + '='*80)
print('FINAL RESULTS SUMMARY')
print('='*80)
print(f"{'T':<8} {'LOLA':<12} {'UCB':<12} {'UCB+LLM':<12} {'Pure LLM':<12} {'E&C':<12}")
print('-'*80)

for _, row in results_df.iterrows():
    T = row['T']
    lola_mean = np.mean(ast.literal_eval(str(row['ucb2_result']))) / T
    ucb_mean = np.mean(ast.literal_eval(str(row['ucb1_result']))) / T
    one_ucb_mean = np.mean(ast.literal_eval(str(row['one_ucb_result']))) / T
    no_bandit_mean = np.mean(ast.literal_eval(str(row['no_bandit_result']))) / T
    ab_mean = np.mean(ast.literal_eval(str(row['ab_result']))) / T

    print(f"{T:<8} {lola_mean:<12.6f} {ucb_mean:<12.6f} {one_ucb_mean:<12.6f} "
          f"{no_bandit_mean:<12.6f} {ab_mean:<12.6f}")

print('='*80)
print('\nSimulation complete!')

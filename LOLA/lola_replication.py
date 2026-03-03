"""
LOLA Replication Script
Replicates the LOLA algorithm using real Upworthy news headline A/B testing data

Benchmark Algorithms:
1. LOLA (LLM-2UCBs) - Proposed Method
2. Standard UCB
3. UCB with LLM Priors
4. Pure LLM (No Bandit)
5. Explore-then-Commit (E&C)
6. Thompson Sampling (TS)
7. LLM-TS (Thompson Sampling with LLM Prior)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import ast
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

class LOLAReplication:
    """Replicate LOLA algorithm and benchmark comparisons"""

    def __init__(self, data_path):
        """Initialize with test data"""
        self.data_path = data_path
        self.test_df = pd.read_csv(data_path)
        self.n_tests = self.test_df.groupby('test_id').ngroups
        print(f"Loaded {len(self.test_df)} headlines from {self.n_tests} tests")

        # Best parameters from the paper
        self.fixed_n_llm = 1000        # Auxiliary sample size for UCB methods
        self.fixed_n_llm_ts = 1200     # Auxiliary sample size for TS
        self.best_z_ucb = 0.08         # Confidence parameter for UCB
        self.best_ratio = 0.2          # Exploration ratio for E&C

        # Thompson Sampling parameters (learned from training data)
        self.alpha_train = 1.38
        self.beta_train = 96.11

    def get_ucb(self, num_trials, num_positive, z_value, t, ctr_llm, n_llm, two_ucb=True):
        """
        Calculate UCB for bandit algorithms

        Parameters:
        -----------
        num_trials : array - number of trials for each arm
        num_positive : array - number of successes for each arm
        z_value : float - confidence parameter
        t : int - current time step
        ctr_llm : array - LLM-predicted CTR for each arm
        n_llm : array - auxiliary sample size for each arm
        two_ucb : bool - if True, use min(standard UCB, LLM UCB) = LOLA
                         if False, use only LLM UCB

        Returns:
        --------
        ucb_values : array - UCB values for each arm
        """
        mean_val = num_positive / num_trials
        ucb = mean_val + z_value * np.sqrt(np.log(t + 2) / num_trials)

        ucb_llm = np.where(n_llm > 0,
                          (ctr_llm * n_llm + num_positive) / (num_trials + n_llm) +
                          z_value * np.sqrt(np.log(t + 2) / (num_trials + n_llm)),
                          ucb)

        if two_ucb:
            return np.minimum(ucb, ucb_llm)  # LOLA: min of both UCBs
        else:
            return ucb_llm

    def simulate_bandit(self, T, ini=1000, two_ucb=True):
        """
        Simulate UCB-based bandit algorithms

        Parameters:
        -----------
        T : int - traffic per headline
        ini : int - auxiliary sample size (n_aux)
        two_ucb : bool - True for LOLA, False for UCB with LLM priors only

        Returns:
        --------
        results : array - total rewards for each test
        """
        rng = np.random.RandomState(42)
        results = []

        for test_id, group in self.test_df.groupby('test_id'):
            n_arms = len(group)
            trials = np.ones(n_arms)
            successes = np.zeros(n_arms)
            ctr_llm = np.array(group['ini_CTR'].values)
            n_llm = ini * np.ones(n_arms)
            total_T = int(n_arms * T)
            rewards = np.zeros(total_T)

            for t in range(total_T):
                ucbs = self.get_ucb(trials, successes, self.best_z_ucb, t, ctr_llm, n_llm, two_ucb)
                chosen_arm = np.argmax(ucbs)
                ctr = group.iloc[chosen_arm]['CTR']
                reward = rng.binomial(1, ctr)
                trials[chosen_arm] += 1
                successes[chosen_arm] += reward
                rewards[t] = reward

            results.append(np.sum(rewards))

        return np.array(results)

    def get_thompson_sampling(self, alpha, beta, n_llm, ctr_llm):
        """
        Sample from Beta distribution for Thompson Sampling

        Parameters:
        -----------
        alpha : array - alpha parameter for each arm
        beta : array - beta parameter for each arm
        n_llm : array - auxiliary sample size
        ctr_llm : array - LLM-predicted CTR

        Returns:
        --------
        samples : array - samples from Beta distribution
        """
        samples = np.random.beta(alpha + ctr_llm * n_llm, beta + (1 - ctr_llm) * n_llm)
        return samples

    def simulate_thompson_sampling(self, T, ini=0):
        """
        Simulate Thompson Sampling algorithms

        Parameters:
        -----------
        T : int - traffic per headline
        ini : int - auxiliary sample size (0 for standard TS, 1200 for LLM-TS)

        Returns:
        --------
        results : array - total rewards for each test
        """
        rng = np.random.RandomState(42)
        results = []

        for test_id, group in self.test_df.groupby('test_id'):
            n_arms = len(group)
            trials = np.ones(n_arms)
            successes = np.zeros(n_arms)
            alpha = self.alpha_train * np.ones(n_arms)
            beta_param = self.beta_train * np.ones(n_arms)
            ctr_llm = np.array(group['ini_CTR'].values)
            ctr_llm = np.maximum(ctr_llm, 0)
            n_llm = ini * np.ones(n_arms)
            total_T = int(n_arms * T)
            rewards = np.zeros(total_T)

            for t in range(total_T):
                ts_samples = self.get_thompson_sampling(alpha, beta_param, n_llm, ctr_llm)
                chosen_arm = np.argmax(ts_samples)
                ctr = group.iloc[chosen_arm]['CTR']
                reward = rng.binomial(1, ctr)
                trials[chosen_arm] += 1
                successes[chosen_arm] += reward
                alpha[chosen_arm] += reward
                beta_param[chosen_arm] += (1 - reward)
                rewards[t] = reward

            results.append(np.sum(rewards))

        return np.array(results)

    def simulate_pure_llm(self, T):
        """
        Simulate Pure LLM (no bandit, always select highest predicted CTR)

        Parameters:
        -----------
        T : int - traffic per headline

        Returns:
        --------
        results : array - total rewards for each test
        """
        rng = np.random.RandomState(42)
        results = []

        for test_id, group in self.test_df.groupby('test_id'):
            n_arms = len(group)
            total_T = int(n_arms * T)
            best_arm = np.argmax(group['ini_CTR'])
            ctr = group.iloc[best_arm]['CTR']
            rewards = rng.binomial(1, ctr, total_T)
            results.append(np.sum(rewards))

        return np.array(results)

    def simulate_explore_then_commit(self, T, ratio=0.2):
        """
        Simulate Explore-then-Commit algorithm

        Parameters:
        -----------
        T : int - traffic per headline
        ratio : float - exploration ratio (default 0.2 = 20%)

        Returns:
        --------
        results : array - total rewards for each test
        """
        rng = np.random.RandomState(42)
        results = []

        for test_id, group in self.test_df.groupby('test_id'):
            n_arms = len(group)
            total_T = int(n_arms * T)
            trials = np.ones(n_arms)
            successes = np.zeros(n_arms)

            explore_T = int(ratio * total_T)
            exploit_T = total_T - explore_T
            rewards = np.zeros(total_T)

            # Exploration phase: uniform random
            chosen_arms = rng.choice(n_arms, explore_T)
            for i, arm in enumerate(chosen_arms):
                ctr = group.iloc[arm]['CTR']
                reward = rng.binomial(1, ctr)
                trials[arm] += 1
                successes[arm] += reward
                rewards[i] = reward

            # Exploitation phase: commit to best arm
            empirical_ctrs = successes / trials
            best_arm = np.argmax(empirical_ctrs)
            ctr = group.iloc[best_arm]['CTR']
            rewards[explore_T:total_T] = rng.binomial(1, ctr, exploit_T)
            results.append(np.sum(rewards))

        return np.array(results)

    def run_all_simulations(self, T_values):
        """
        Run simulations for all benchmark algorithms across different T values

        Parameters:
        -----------
        T_values : list - traffic per headline values to test

        Returns:
        --------
        results_df : DataFrame - simulation results for all algorithms
        """
        results_list = []

        for T in T_values:
            print(f"\n{'='*50}")
            print(f"Running simulations for T = {T}")
            print(f"{'='*50}")

            result_row = {'T': T}

            # Standard UCB (no LLM prior)
            print(f"  [1/7] Standard UCB...")
            result_row['ucb1_result'] = self.simulate_bandit(T, ini=0, two_ucb=True).tolist()

            # LOLA (LLM-2UCBs)
            print(f"  [2/7] LOLA (LLM-2UCBs)...")
            result_row['ucb2_result'] = self.simulate_bandit(T, ini=self.fixed_n_llm, two_ucb=True).tolist()

            # UCB with LLM priors only
            print(f"  [3/7] UCB with LLM priors...")
            result_row['one_ucb_result'] = self.simulate_bandit(T, ini=self.fixed_n_llm, two_ucb=False).tolist()

            # Pure LLM (no bandit)
            print(f"  [4/7] Pure LLM...")
            result_row['no_bandit_result'] = self.simulate_pure_llm(T).tolist()

            # Explore-then-Commit
            print(f"  [5/7] Explore-then-Commit...")
            result_row['ab_result'] = self.simulate_explore_then_commit(T, self.best_ratio).tolist()

            # Thompson Sampling
            print(f"  [6/7] Thompson Sampling...")
            result_row['ts_result'] = self.simulate_thompson_sampling(T, ini=0).tolist()

            # LLM-TS
            print(f"  [7/7] LLM-TS...")
            result_row['ts_llm_result'] = self.simulate_thompson_sampling(T, ini=self.fixed_n_llm_ts).tolist()

            results_list.append(result_row)
            print(f"  Completed T = {T}")

        return pd.DataFrame(results_list)

    def plot_comparison(self, results_df, save_path='lola_comparison.pdf'):
        """
        Generate comparison plot similar to Figure 2 in the paper

        Parameters:
        -----------
        results_df : DataFrame - simulation results
        save_path : str - path to save the plot
        """
        plt.figure(figsize=(10, 6))
        results_df['T'] = pd.to_numeric(results_df['T'], errors='coerce')

        # Define algorithms and their visual properties
        algorithms = [
            ('ucb2_result', 'LOLA (LLM-2UCBs)', 'o', '-', 'blue'),
            ('one_ucb_result', 'UCB with LLM priors', 'o', '-', 'black'),
            ('ucb1_result', 'UCB', 's', '--', 'green'),
            ('no_bandit_result', 'Pure LLM', 'D', '-.', 'red'),
            ('ab_result', 'E&C', '^', ':', 'purple')
        ]

        for col, label, marker, linestyle, color in algorithms:
            results_df[col] = results_df[col].apply(ast.literal_eval)
            results_df[f'{label}_mean'] = results_df.apply(
                lambda row: np.mean(np.array([row[col]]) / row['T']), axis=1)
            plt.plot(results_df['T'], results_df[f'{label}_mean'],
                    marker=marker, linestyle=linestyle, color=color, label=label)

        plt.xlabel(r'Traffic/Impressions per headline in tests $\tau$')
        plt.ylabel('Average clicks per test per period')
        plt.xticks([50, 100, 200, 400, 600, 800, 1000])
        plt.xlim([50, 1000])
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.show()

    def plot_comparison_with_ts(self, results_df, save_path='lola_comparison_ts.pdf'):
        """
        Generate comparison plot including Thompson Sampling variants

        Parameters:
        -----------
        results_df : DataFrame - simulation results
        save_path : str - path to save the plot
        """
        plt.figure(figsize=(10, 6))
        results_df['T'] = pd.to_numeric(results_df['T'], errors='coerce')

        # Define algorithms and their visual properties
        algorithms = [
            ('ucb2_result', 'LOLA (LLM-2UCBs)', 'o', '-', 'blue'),
            ('ucb1_result', 'UCB', 's', '--', 'green'),
            ('no_bandit_result', 'Pure LLM', 'D', '-.', 'red'),
            ('ab_result', 'E&C', '^', ':', 'purple'),
            ('ts_llm_result', 'LLM-TS', '<', '-', 'brown'),
            ('ts_result', 'TS', 'v', '-', 'black')
        ]

        for col, label, marker, linestyle, color in algorithms:
            results_df[col] = results_df[col].apply(ast.literal_eval)
            results_df[f'{label}_mean'] = results_df.apply(
                lambda row: np.mean(np.array([row[col]]) / row['T']), axis=1)
            plt.plot(results_df['T'], results_df[f'{label}_mean'],
                    marker=marker, linestyle=linestyle, color=color, label=label)

        plt.xlabel(r'Traffic/Impressions per headline in tests $\tau$')
        plt.ylabel('Average clicks per test per period')
        plt.xticks([50, 100, 200, 400, 600, 800, 1000])
        plt.xlim([50, 1000])
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
        plt.show()

    def generate_comparison_table(self, results_df):
        """
        Generate statistical comparison table with mean differences and p-values

        Parameters:
        -----------
        results_df : DataFrame - simulation results

        Returns:
        --------
        comparison_df : DataFrame - comparison statistics
        """
        # Map labels to column names
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
                data1 = np.array([row[col1]]) / row['T']
                data2 = np.array([row[col2]]) / row['T']

                mean_diff = np.mean(data1[0] - data2[0]) / np.mean(data2[0]) * 100
                t_stat, p_value = ttest_rel(data1[0], data2[0])

                mean_diffs.append(mean_diff)
                p_values.append(p_value)

            comparison_df[f'{label1} vs {label2}_mean_diff'] = mean_diffs
            comparison_df[f'{label1} vs {label2}_p_value'] = p_values

        return comparison_df

    def print_summary_statistics(self, results_df):
        """
        Print summary statistics for all algorithms

        Parameters:
        -----------
        results_df : DataFrame - simulation results
        """
        print("\n" + "="*80)
        print("LOLA REPLICATION SUMMARY")
        print("="*80)

        algorithms = [
            ('ucb2_result', 'LOLA (LLM-2UCBs)'),
            ('ucb1_result', 'UCB'),
            ('one_ucb_result', 'UCB with LLM priors'),
            ('no_bandit_result', 'Pure LLM'),
            ('ab_result', 'E&C'),
            ('ts_result', 'TS'),
            ('ts_llm_result', 'LLM-TS')
        ]

        print("\nAverage clicks per test per period:")
        print("-" * 80)
        print(f"{'T':<8} {'LOLA':<12} {'UCB':<12} {'UCB+LLM':<12} {'Pure LLM':<12} {'E&C':<12}")
        print("-" * 80)

        for _, row in results_df.iterrows():
            T = row['T']
            lola_mean = np.mean(np.array([row['ucb2_result']]) / T)
            ucb_mean = np.mean(np.array([row['ucb1_result']]) / T)
            one_ucb_mean = np.mean(np.array([row['one_ucb_result']]) / T)
            no_bandit_mean = np.mean(np.array([row['no_bandit_result']]) / T)
            ab_mean = np.mean(np.array([row['ab_result']]) / T)

            print(f"{T:<8} {lola_mean:<12.6f} {ucb_mean:<12.6f} {one_ucb_mean:<12.6f} "
                  f"{no_bandit_mean:<12.6f} {ab_mean:<12.6f}")

        print("\n" + "="*80)


def main():
    """Main execution function"""
    # Set up paths
    base_path = Path(r"D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication")
    data_path = base_path / "LLM_News" / "LOLA - Regret Minimize" / "LoRA CTR.csv"
    output_dir = base_path / "results"
    output_dir.mkdir(exist_ok=True)

    # Initialize replication
    print("Initializing LOLA Replication...")
    print(f"Data path: {data_path}")
    replication = LOLAReplication(str(data_path))

    # Run simulations for all T values
    T_values = [50, 100, 200, 400, 600, 800, 1000]
    print(f"\nRunning simulations for T values: {T_values}")
    results_df = replication.run_all_simulations(T_values)

    # Save results
    results_path = output_dir / "simulation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    replication.plot_comparison(results_df, save_path=output_dir / "lola_comparison.pdf")
    replication.plot_comparison_with_ts(results_df, save_path=output_dir / "lola_comparison_ts.pdf")

    # Generate comparison table
    comparison_df = replication.generate_comparison_table(results_df)
    comparison_path = output_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to {comparison_path}")

    # Print summary statistics
    replication.print_summary_statistics(results_df)

    print("\n" + "="*80)
    print("REPLICATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"- Simulation results: {results_path}")
    print(f"- Comparison plot: {output_dir / 'lola_comparison.pdf'}")
    print(f"- Comparison with TS: {output_dir / 'lola_comparison_ts.pdf'}")
    print(f"- Comparison table: {comparison_path}")

    return results_df, comparison_df


if __name__ == "__main__":
    results_df, comparison_df = main()

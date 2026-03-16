"""
LOLA Algorithm Implementations
Based on Ye et al. (2025) "LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments"

This module contains:
- LLM-2UCBs: Main algorithm for regret minimization
- LLM-TS: Thompson Sampling variant
- LLM-BAI: Best Arm Identification variant
- Benchmark algorithms: Pure UCB, Pure TS, E&C, Pure LLM
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LLM-2UCBs: Main Algorithm (Algorithm 1 from paper)
# =============================================================================

class LLM_2UCBs:
    """
    LLM-Assisted 2-Upper Confidence Bounds Algorithm.
    
    From Ye et al. (2025), Algorithm 1.
    
    The key innovation is using two UCBs:
    - U¹: Standard UCB using observed data (unbiased but high variance)
    - U²: UCB incorporating LLM predictions as auxiliary samples
    - Final decision: argmax min{U¹, U²}
    
    This balances the risk of LLM overestimation against exploration efficiency.
    """
    
    def __init__(self, n_arms: int, predicted_CTR: np.ndarray, 
                 naux: int = 1000, alpha: float = 0.08):
        """
        Initialize LLM-2UCBs.
        
        Parameters
        ----------
        n_arms : int
            Number of arms (headlines) in the bandit
        predicted_CTR : np.ndarray
            LLM-predicted CTR for each arm (μ̄ᵏᵃᵘˣ), shape (n_arms,)
        naux : int, default=1000
            LLM's equivalent auxiliary sample size.
            - naux = 0: Equivalent to standard UCB
            - naux = ∞: Equivalent to Pure LLM (greedy)
        alpha : float, default=0.08
            Upper confidence bound control parameter
        """
        self.n_arms = n_arms
        self.predicted_CTR = np.array(predicted_CTR).flatten()
        self.naux = naux
        self.alpha = alpha
        
        # Initialize counters
        self.n_k = np.ones(n_arms, dtype=np.float64)  # Number of impressions
        self.c_k = np.zeros(n_arms, dtype=np.float64)  # Number of clicks
        self.mu_hat_k = np.zeros(n_arms, dtype=np.float64)  # Empirical CTR
        self.t = 0  # Time step
        
        # History for analysis
        self.arm_history = []
        self.reward_history = []
        
    def select_arm(self) -> int:
        """
        Select arm using 2-UCBs rule: at = argmax min{U¹, U²}.
        
        Returns
        -------
        int
            Selected arm index
        """
        t = self.t + 1
        
        # First UCB: Standard UCB using observed data
        # U¹_k = μ̂_k + α * sqrt(log(t) / n_k)
        U1 = self.mu_hat_k + self.alpha * np.sqrt(np.log(t) / self.n_k)
        
        # Second UCB: UCB incorporating LLM predictions
        # Mean estimator with auxiliary samples:
        # (c_k + μ̄ᵃᵘˣ_k * naux) / (n_k + naux)
        mean_with_aux = (self.c_k + self.predicted_CTR * self.naux) / (self.n_k + self.naux)
        U2 = mean_with_aux + self.alpha * np.sqrt(np.log(t) / (self.n_k + self.naux))
        
        # Select arm that maximizes the minimum of two UCBs
        ucb_values = np.minimum(U1, U2)
        selected_arm = np.argmax(ucb_values)
        
        return int(selected_arm)
    
    def update(self, arm: int, reward: float) -> None:
        """
        Update after observing reward.
        
        Parameters
        ----------
        arm : int
            Selected arm index
        reward : float
            Observed reward (0 or 1 for clicks)
        """
        self.n_k[arm] += 1
        self.c_k[arm] += reward
        self.mu_hat_k[arm] = self.c_k[arm] / self.n_k[arm]
        self.t += 1
        
        self.arm_history.append(arm)
        self.reward_history.append(reward)
    
    def run(self, real_CTR: np.ndarray, T: int) -> int:
        """
        Run the algorithm for T rounds.
        
        Parameters
        ----------
        real_CTR : np.ndarray
            True CTR for each arm (for simulation)
        T : int
            Time horizon
            
        Returns
        -------
        int
            Total clicks accumulated
        """
        real_CTR = np.array(real_CTR).flatten()
        total_clicks = 0
        
        for _ in range(T):
            arm = self.select_arm()
            reward = np.random.binomial(1, real_CTR[arm])
            self.update(arm, reward)
            total_clicks += reward
        
        return total_clicks
    
    def reset(self) -> None:
        """Reset the algorithm state."""
        self.n_k = np.ones(self.n_arms, dtype=np.float64)
        self.c_k = np.zeros(self.n_arms, dtype=np.float64)
        self.mu_hat_k = np.zeros(self.n_arms, dtype=np.float64)
        self.t = 0
        self.arm_history = []
        self.reward_history = []


# =============================================================================
# LLM-Assisted Thompson Sampling (Algorithm 3 from Appendix G.2)
# =============================================================================

class LLM_TS:
    """
    LLM-Assisted Thompson Sampling Algorithm.
    
    From Ye et al. (2025), Algorithm 3 (Appendix G.2).
    
    Extends standard TS by initializing Beta priors with LLM pseudo-samples:
    - αₖ¹ = α₀ + naux * μ̄ᵏᵃᵘˣ
    - βₖ¹ = β₀ + naux * (1 - μ̄ᵏᵃᵘˣ)
    
    Note: UCB-based algorithms tend to outperform TS in this setting due to
    TS's over-exploration tendency (Min et al., 2020).
    """
    
    def __init__(self, n_arms: int, predicted_CTR: np.ndarray,
                 naux: int = 1200, alpha_0: float = 1.38, beta_0: float = 96.11):
        """
        Initialize LLM-TS.
        
        Parameters
        ----------
        n_arms : int
            Number of arms
        predicted_CTR : np.ndarray
            LLM-predicted CTR for each arm
        naux : int, default=1200
            LLM's equivalent auxiliary sample size
        alpha_0 : float, default=1.38
            Beta prior α parameter (computed from training data CTR distribution)
        beta_0 : float, default=96.11
            Beta prior β parameter (computed from training data CTR distribution)
        """
        self.n_arms = n_arms
        self.predicted_CTR = np.array(predicted_CTR).flatten()
        self.naux = naux
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        
        # Initialize Beta priors with LLM pseudo-samples
        # αₖ¹ = α₀ + naux * μ̄ᵏᵃᵘˣ
        # βₖ¹ = β₀ + naux * (1 - μ̄ᵏᵃᵘˣ)
        self.alpha_k = alpha_0 + naux * self.predicted_CTR
        self.beta_k = beta_0 + naux * (1 - self.predicted_CTR)
        
        # History
        self.arm_history = []
        self.reward_history = []
        
    def select_arm(self) -> int:
        """
        Select arm by sampling from Beta posteriors.
        
        Returns
        -------
        int
            Selected arm index
        """
        samples = np.array([np.random.beta(self.alpha_k[k], self.beta_k[k]) 
                           for k in range(self.n_arms)])
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float) -> None:
        """Update Beta posterior after observing reward."""
        self.alpha_k[arm] += reward
        self.beta_k[arm] += (1 - reward)
        
        self.arm_history.append(arm)
        self.reward_history.append(reward)
    
    def run(self, real_CTR: np.ndarray, T: int) -> int:
        """Run for T rounds."""
        real_CTR = np.array(real_CTR).flatten()
        total_clicks = 0
        
        for _ in range(T):
            arm = self.select_arm()
            reward = np.random.binomial(1, real_CTR[arm])
            self.update(arm, reward)
            total_clicks += reward
        
        return total_clicks
    
    def reset(self) -> None:
        """Reset the algorithm state."""
        self.alpha_k = self.alpha_0 + self.naux * self.predicted_CTR
        self.beta_k = self.beta_0 + self.naux * (1 - self.predicted_CTR)
        self.arm_history = []
        self.reward_history = []


# =============================================================================
# LLM-BAI: Best Arm Identification (Algorithm 2 from Appendix G.1)
# =============================================================================

class LLM_BAI:
    """
    LLM-Assisted Best Arm Identification.
    
    From Ye et al. (2025), Algorithm 2 (Appendix G.1).
    
    Based on (ST)² algorithm from Mason et al. (2020), adapted to use
    LLM predictions for initialization.
    
    Returns a set of "good" arms with CTR > μ* - γ with probability ≥ 1 - δ.
    """
    
    def __init__(self, real_CTR: np.ndarray, predicted_CTR: np.ndarray,
                 naux: int = 300, epsilon: float = 0, delta: float = 0.2,
                 gamma: float = 0.005, c_Phi: float = 0.0015, max_pulls: int = 300):
        """
        Initialize LLM-BAI.
        
        Parameters
        ----------
        real_CTR : np.ndarray
            True CTR for simulation
        predicted_CTR : np.ndarray
            LLM-predicted CTR
        naux : int, default=300
            Initial pulls based on LLM prediction ranking
        delta : float, default=0.2
            Failure probability (confidence level = 1 - δ)
        gamma : float, default=0.005
            Threshold for "good" arms (μ* - γ = best CTR - 0.5%)
        c_Phi : float, default=0.0015
            Confidence interval scaling constant
        max_pulls : int, default=300
            Maximum number of pulls allowed
        """
        self.real_CTR = np.array(real_CTR).flatten()
        self.predicted_CTR = np.array(predicted_CTR).flatten()
        self.n_arms = len(self.real_CTR)
        self.naux = naux
        self.delta = delta
        self.gamma = gamma
        self.c_Phi = c_Phi
        self.max_pulls = max_pulls
        
        # Initialize with LLM predictions
        # Start with naux virtual pulls based on LLM prediction
        self.T_k = np.ones(self.n_arms) * naux  # Initial pulls
        self.mu_bar_k = self.predicted_CTR.copy()  # CTR estimates
        
        # Candidate set
        self.C = list(range(self.n_arms))
        
    def confidence_interval(self, T_k: float) -> float:
        """Compute confidence interval width: c_Phi * sqrt(log(1/δ) / T_k)."""
        return self.c_Phi * np.sqrt(np.log(1/self.delta) / max(T_k, 1))
    
    def run(self) -> Tuple[bool, int, List[int]]:
        """
        Run BAI algorithm.
        
        Returns
        -------
        success : bool
            Whether all selected arms have CTR > μ* - γ
        n_pulls : int
            Total number of pulls
        selected_arms : List[int]
            The set of good arms identified
        """
        n_pulls = int(self.n_arms * self.naux)
        
        while len(self.C) > 1 and n_pulls < self.max_pulls:
            # Pull each arm in candidate set once
            for k in self.C:
                reward = np.random.binomial(1, self.real_CTR[k])
                self.T_k[k] += 1
                # Update CTR estimate (running average)
                old_estimate = self.mu_bar_k[k]
                n = self.T_k[k]
                self.mu_bar_k[k] = old_estimate * (n - 1) / n + reward / n
                n_pulls += 1
            
            # Compute confidence intervals
            CI_lower = [self.mu_bar_k[k] - self.confidence_interval(self.T_k[k]) 
                       for k in self.C]
            CI_upper = [self.mu_bar_k[k] + self.confidence_interval(self.T_k[k]) 
                       for k in self.C]
            
            # Eliminate suboptimal arms
            # An arm is eliminated if its upper bound < max lower bound - γ
            max_lower = max(CI_lower)
            self.C = [k for i, k in enumerate(self.C) 
                     if CI_upper[i] >= max_lower - self.gamma]
            
            # Safety: keep at least one arm
            if len(self.C) == 0:
                self.C = [np.argmax(self.mu_bar_k)]
                break
        
        # Check success: all selected arms have CTR > μ* - γ
        mu_star = np.max(self.real_CTR)
        success = all(self.real_CTR[k] >= mu_star - self.gamma for k in self.C)
        
        return success, n_pulls, self.C.copy()


# =============================================================================
# Benchmark Algorithms
# =============================================================================

class Pure_UCB:
    """
    Standard UCB algorithm without LLM priors.
    
    This is LOLA with naux = 0 (no trust in LLM predictions).
    
    From Auer et al. (2002), asymptotically optimal for stochastic bandits.
    """
    
    def __init__(self, n_arms: int, alpha: float = 0.08):
        """
        Initialize Pure UCB.
        
        Parameters
        ----------
        n_arms : int
            Number of arms
        alpha : float, default=0.08
            Confidence control parameter
        """
        self.n_arms = n_arms
        self.alpha = alpha
        self.n_k = np.ones(n_arms, dtype=np.float64)
        self.c_k = np.zeros(n_arms, dtype=np.float64)
        self.mu_hat_k = np.zeros(n_arms, dtype=np.float64)
        self.t = 0
        
    def select_arm(self) -> int:
        """Select arm with highest UCB."""
        t = self.t + 1
        ucb = self.mu_hat_k + self.alpha * np.sqrt(np.log(t) / self.n_k)
        return int(np.argmax(ucb))
    
    def update(self, arm: int, reward: float) -> None:
        """Update after observing reward."""
        self.n_k[arm] += 1
        self.c_k[arm] += reward
        self.mu_hat_k[arm] = self.c_k[arm] / self.n_k[arm]
        self.t += 1
    
    def run(self, real_CTR: np.ndarray, T: int) -> int:
        """Run for T rounds."""
        real_CTR = np.array(real_CTR).flatten()
        total_clicks = 0
        
        for _ in range(T):
            arm = self.select_arm()
            reward = np.random.binomial(1, real_CTR[arm])
            self.update(arm, reward)
            total_clicks += reward
        
        return total_clicks
    
    def reset(self) -> None:
        """Reset the algorithm state."""
        self.n_k = np.ones(self.n_arms, dtype=np.float64)
        self.c_k = np.zeros(self.n_arms, dtype=np.float64)
        self.mu_hat_k = np.zeros(self.n_arms, dtype=np.float64)
        self.t = 0


class Pure_TS:
    """
    Standard Thompson Sampling without LLM priors.
    
    Uses Beta(α₀, β₀) prior for all arms, updated with observed data.
    """
    
    def __init__(self, n_arms: int, alpha_0: float = 1.38, beta_0: float = 96.11):
        """
        Initialize Pure TS.
        
        Parameters
        ----------
        n_arms : int
            Number of arms
        alpha_0 : float, default=1.38
            Prior α parameter (from training data CTR distribution)
        beta_0 : float, default=96.11
            Prior β parameter (from training data CTR distribution)
        """
        self.n_arms = n_arms
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.alpha_k = np.ones(n_arms) * alpha_0
        self.beta_k = np.ones(n_arms) * beta_0
    
    def select_arm(self) -> int:
        """Select arm by sampling from Beta posteriors."""
        samples = np.array([np.random.beta(self.alpha_k[k], self.beta_k[k]) 
                           for k in range(self.n_arms)])
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float) -> None:
        """Update Beta posterior."""
        self.alpha_k[arm] += reward
        self.beta_k[arm] += (1 - reward)
    
    def run(self, real_CTR: np.ndarray, T: int) -> int:
        """Run for T rounds."""
        real_CTR = np.array(real_CTR).flatten()
        total_clicks = 0
        
        for _ in range(T):
            arm = self.select_arm()
            reward = np.random.binomial(1, real_CTR[arm])
            self.update(arm, reward)
            total_clicks += reward
        
        return total_clicks
    
    def reset(self) -> None:
        """Reset the algorithm state."""
        self.alpha_k = np.ones(self.n_arms) * self.alpha_0
        self.beta_k = np.ones(self.n_arms) * self.beta_0


def run_EandC(real_CTR: np.ndarray, exploration_ratio: float = 0.2, 
              T: Optional[int] = None) -> int:
    """
    Explore & Commit (Standard A/B Test).
    
    This was the status quo at Upworthy during the data collection period.
    The firm runs an A/B test for a fixed proportion of periods, then commits
    to the best-performing arm for the remaining traffic.
    
    Parameters
    ----------
    real_CTR : np.ndarray
        True CTR for each arm
    exploration_ratio : float, default=0.2
        Proportion of periods for exploration (0.2 = 20%)
    T : int, optional
        Total time horizon. If None, computed as n_arms * 100
        
    Returns
    -------
    int
        Total clicks accumulated
    """
    real_CTR = np.array(real_CTR).flatten()
    n_arms = len(real_CTR)
    
    if T is None:
        T = n_arms * 100
    
    explore_T = int(T * exploration_ratio)
    pulls_per_arm = max(1, explore_T // n_arms)
    
    # Exploration phase: equal allocation
    observed_clicks = np.zeros(n_arms)
    for arm in range(n_arms):
        for _ in range(pulls_per_arm):
            observed_clicks[arm] += np.random.binomial(1, real_CTR[arm])
    
    # Select best arm based on empirical CTR
    empirical_CTR = observed_clicks / pulls_per_arm
    best_arm = np.argmax(empirical_CTR)
    
    # Exploitation phase
    remaining_T = T - n_arms * pulls_per_arm
    if remaining_T > 0:
        exploitation_clicks = np.random.binomial(remaining_T, real_CTR[best_arm])
    else:
        exploitation_clicks = 0
    
    return int(observed_clicks.sum() + exploitation_clicks)


def run_Pure_LLM(real_CTR: np.ndarray, predicted_CTR: np.ndarray,
                 T: Optional[int] = None) -> int:
    """
    Pure LLM approach: Select arm with highest predicted CTR for all periods.
    
    Equivalent to LOLA with naux → ∞ (full trust in LLM).
    This is a fully greedy policy based on LLM predictions.
    
    Parameters
    ----------
    real_CTR : np.ndarray
        True CTR for each arm (for simulation)
    predicted_CTR : np.ndarray
        LLM-predicted CTR for each arm
    T : int, optional
        Total time horizon. If None, computed as n_arms * 100
        
    Returns
    -------
    int
        Total clicks accumulated
    """
    real_CTR = np.array(real_CTR).flatten()
    predicted_CTR = np.array(predicted_CTR).flatten()
    n_arms = len(real_CTR)
    
    if T is None:
        T = n_arms * 100
    
    best_arm = np.argmax(predicted_CTR)
    return int(np.random.binomial(T, real_CTR[best_arm]))


def run_UCB_with_LLM_priors(real_CTR: np.ndarray, predicted_CTR: np.ndarray,
                            naux: int = 1000, alpha: float = 0.08,
                            T: Optional[int] = None) -> int:
    """
    UCB with LLM priors using only U² (not min{U¹, U²}).
    
    This demonstrates why using min{U¹, U²} is better than just U².
    If LLM overestimates a poor arm's CTR, it takes many rounds to correct.
    
    Parameters
    ----------
    real_CTR : np.ndarray
        True CTR for each arm
    predicted_CTR : np.ndarray
        LLM-predicted CTR
    naux : int, default=1000
        Auxiliary sample size
    alpha : float, default=0.08
        Confidence control parameter
    T : int, optional
        Time horizon
        
    Returns
    -------
    int
        Total clicks accumulated
    """
    real_CTR = np.array(real_CTR).flatten()
    predicted_CTR = np.array(predicted_CTR).flatten()
    n_arms = len(real_CTR)
    
    if T is None:
        T = n_arms * 100
    
    n_k = np.ones(n_arms)
    c_k = np.zeros(n_arms)
    
    total_clicks = 0
    
    for t in range(1, T + 1):
        # Only use U² (UCB with LLM prior)
        mean_with_aux = (c_k + predicted_CTR * naux) / (n_k + naux)
        U2 = mean_with_aux + alpha * np.sqrt(np.log(t) / (n_k + naux))
        
        arm = np.argmax(U2)
        reward = np.random.binomial(1, real_CTR[arm])
        
        n_k[arm] += 1
        c_k[arm] += reward
        total_clicks += reward
    
    return total_clicks


# =============================================================================
# Utility Functions
# =============================================================================

def compute_ts_prior(train_CTR: np.ndarray) -> Tuple[float, float]:
    """
    Compute Thompson Sampling prior parameters from training data.
    
    Uses method of moments to fit Beta distribution to CTR distribution.
    
    Parameters
    ----------
    train_CTR : np.ndarray
        Array of CTR values from training data
        
    Returns
    -------
    alpha_0 : float
        Prior α parameter
    beta_0 : float
        Prior β parameter
    """
    mean_ctr = np.mean(train_CTR)
    var_ctr = np.var(train_CTR)
    
    # Method of moments for Beta distribution
    # mean = α / (α + β)
    # var = αβ / ((α + β)²(α + β + 1))
    common_term = (mean_ctr * (1 - mean_ctr) / var_ctr) - 1
    alpha_0 = mean_ctr * common_term
    beta_0 = (1 - mean_ctr) * common_term
    
    return alpha_0, beta_0


def run_all_algorithms(real_CTR: np.ndarray, predicted_CTR: np.ndarray,
                       T: int, naux_ucb: int = 1000, alpha: float = 0.08,
                       naux_ts: int = 1200, alpha_0: float = 1.38,
                       beta_0: float = 96.11) -> Dict[str, int]:
    """
    Run all algorithms and return results for comparison.
    
    Parameters
    ----------
    real_CTR : np.ndarray
        True CTR for each arm
    predicted_CTR : np.ndarray
        LLM-predicted CTR
    T : int
        Time horizon
    naux_ucb : int, default=1000
        naux for LLM-2UCBs
    alpha : float, default=0.08
        Confidence control parameter
    naux_ts : int, default=1200
        naux for LLM-TS
    alpha_0, beta_0 : float
        TS prior parameters
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping algorithm name to total clicks
    """
    real_CTR = np.array(real_CTR).flatten()
    predicted_CTR = np.array(predicted_CTR).flatten()
    n_arms = len(real_CTR)
    
    results = {}
    
    # LLM-2UCBs (LOLA)
    algo = LLM_2UCBs(n_arms, predicted_CTR, naux_ucb, alpha)
    results['LLM-2UCBs'] = algo.run(real_CTR, T)
    
    # Pure UCB
    algo = Pure_UCB(n_arms, alpha)
    results['UCB'] = algo.run(real_CTR, T)
    
    # UCB with LLM priors (only U²)
    results['UCB_LLM_prior'] = run_UCB_with_LLM_priors(real_CTR, predicted_CTR, naux_ucb, alpha, T)
    
    # LLM-TS
    algo = LLM_TS(n_arms, predicted_CTR, naux_ts, alpha_0, beta_0)
    results['LLM-TS'] = algo.run(real_CTR, T)
    
    # Pure TS
    algo = Pure_TS(n_arms, alpha_0, beta_0)
    results['TS'] = algo.run(real_CTR, T)
    
    # Pure LLM
    results['Pure_LLM'] = run_Pure_LLM(real_CTR, predicted_CTR, T)
    
    # E&C
    results['E&C'] = run_EandC(real_CTR, 0.2, T)
    
    return results


if __name__ == "__main__":
    # Test the algorithms
    np.random.seed(42)
    
    # Create a simple test case
    n_arms = 4
    real_CTR = np.array([0.015, 0.018, 0.012, 0.020])  # Arm 4 is best
    predicted_CTR = np.array([0.016, 0.017, 0.013, 0.019])  # LLM correctly identifies best
    
    T = 400
    
    print("Testing LOLA algorithms...")
    print(f"Real CTR: {real_CTR}")
    print(f"Predicted CTR: {predicted_CTR}")
    print(f"Best arm: {np.argmax(real_CTR)} (real), {np.argmax(predicted_CTR)} (predicted)")
    print()
    
    results = run_all_algorithms(real_CTR, predicted_CTR, T)
    
    print("Results (total clicks out of 400):")
    for method, clicks in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {method:20s}: {clicks:3d} ({clicks/T:.4f} per period)")

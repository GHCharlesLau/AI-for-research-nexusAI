# LOLA Benchmark Algorithms Summary

Based on "LOLA: LLM-assisted Online Learning Algorithm for Content Experiments" (Ye et al., 2025)

## Problem Setup

- **K-armed bandit problem**: Each test has K headlines (arms), each with unknown click-through rate (CTR)
- **Goal**: Maximize total clicks (regret minimization) or identify the best arm (best arm identification)
- **Data**: Real Upworthy news headline A/B testing data with actual CTR values

## Benchmark Algorithms

### 1. LOLA (LLM-2UCBs) - Proposed Method

**Core Idea**: Combine standard UCB with LLM-prior UCB by taking the minimum

**Algorithm**:
```
U_t^k(s) = μ̂_t^k(s) + z·√(ln(t+2) / n_t^k(s))              # Standard UCB
Ũ_t^k(s) = μ̃_t^k(s) + z·√(ln(t+2) / (n_t^k(s) + n_aux))     # LLM UCB

LOLA Selection: argmin_k {U_t^k(s), Ũ_t^k(s)}
```

**Key Parameters**:
- n_aux = 1000 (auxiliary sample size)
- z = 0.08 (confidence parameter)
- Prior: LLM-predicted CTR from fine-tuned Llama-3

### 2. Standard UCB

**Algorithm**: UCB without LLM prior
```
U_t^k(s) = μ̂_t^k(s) + z·√(ln(t+2) / n_t^k(s))
Selection: argmax_k U_t^k(s)
```

**Key Parameters**:
- z = 0.08

### 3. UCB with LLM Priors

**Algorithm**: Only uses LLM UCB (not the minimum)
```
Ũ_t^k(s) = μ̃_t^k(s) + z·√(ln(t+2) / (n_t^k(s) + n_aux))
Selection: argmax_k Ũ_t^k(s)
```

**Key Parameters**:
- n_aux = 1000
- z = 0.08

### 4. Pure LLM (No Bandit)

**Algorithm**: Always select the arm with highest LLM-predicted CTR
```
Selection: argmax_k CTR_LLM^k(s)
No exploration/exploration trade-off
```

### 5. Explore-then-Commit (E&C)

**Algorithm**: Uniform random exploration, then commit to best arm
```
Phase 1 (Exploration): 20% of traffic - uniform random selection
Phase 2 (Exploitation): 80% of traffic - select arm with highest empirical CTR
```

### 6. Thompson Sampling (TS)

**Algorithm**: Bayesian approach with Beta conjugate prior
```
Prior: θ_k ~ Beta(α, β)
Posterior: θ_k | data ~ Beta(α + successes, β + failures)
Sample: θ̂_k ~ Beta(α + s_t^k, β + f_t^k)
Selection: argmax_k θ̂_k
```

**Key Parameters**:
- α = 1.38 (learned from training data)
- β = 96.11 (learned from training data)

### 7. LLM-TS (Thompson Sampling with LLM Prior)

**Algorithm**: Thompson Sampling with LLM prior as pseudo-observations
```
θ̂_k ~ Beta(α + CTR_LLM^k·n_aux, β + (1-CTR_LLM^k)·n_aux)
Selection: argmax_k θ̂_k
```

**Key Parameters**:
- n_aux = 1200 (optimal for TS variant)
- α = 1.38, β = 96.11

## Algorithm Comparison Table

| Algorithm | Uses LLM? | Uses Data? | Exploration | Key Innovation |
|-----------|-----------|------------|-------------|----------------|
| **LOLA** | ✓ | ✓ | Adaptive | Min of two UCBs |
| UCB | ✗ | ✓ | Adaptive | Classic UCB |
| UCB+LLM | ✓ | ✓ | Adaptive | LLM prior only |
| Pure LLM | ✓ | ✗ | None | No exploration |
| E&C | ✗ | ✓ | Fixed 20% | Simple two-phase |
| TS | ✗ | ✓ | Adaptive | Bayesian sampling |
| LLM-TS | ✓ | ✓ | Adaptive | LLM as pseudo-data |

## Performance Results (from Paper)

### Regret Minimization
- **LOLA vs UCB**: +2-7% improvement (p < 0.05)
- **LOLA vs Pure LLM**: +1-5% (p < 0.05 for T ≥ 200)
- **LOLA vs E&C**: +5-8% (p < 0.05)
- **LOLA vs TS**: +1-2% (p < 0.05)

### Key Insights
1. **LOLA dominates** all baselines across different traffic levels (τ = 50-1000)
2. **LLM priors help** when n_aux is properly tuned (1000-1800 optimal)
3. **Two-UCB approach** (taking minimum) is better than using LLM UCB alone
4. **LOLA achieves ~95%** of the optimal (oracle) performance

## Sensitivity Analysis

### n_aux (auxiliary sample size)
- n_aux = 0: Same as standard UCB
- n_aux = 200-600: Moderate improvement
- n_aux = 1000: **Optimal for UCB-based methods**
- n_aux = 1800-5000: Diminishing returns

### z (confidence parameter)
- z = 0.08: Optimal across all methods
- Too small: Insufficient exploration
- Too large: Too much exploration

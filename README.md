# Enhanced-Gibbs-Sampling-with-Metropolis-Hastings-Steps
This project explores the application of Bayesian inference techniques through the implementation of the Metropolis-Hastings algorithm and enhanced Gibbs sampling. The goal is to demonstrate how these methods can be used to sample from complex probability distributions and to perform Bayesian analysis in hierarchical models

The project is divided into two main parts:

1. Sampling from a Gamma Distribution Using Metropolis-Hastings:
The Metropolis-Hastings algorithm, a cornerstone of Markov Chain Monte Carlo (MCMC) methods, is employed to sample from a gamma distribution. This part of the project serves as an introductory
example of using MCMC techniques to obtain random samples from a probability distribution for which direct sampling is not straightforward. The implementation is carried out in Python,
showcasing how to propose new states, calculate acceptance probabilities, and decide whether to accept or reject proposed states based on these probabilities. The example culminates in visualizing the sampled values to assess the effectiveness of the sampling process.

## Metropolis-Hastings for Gamma Distribution
The Metropolis-Hastings algorithm is a cornerstone of Markov Chain Monte Carlo (MCMC) methods, allowing for sampling from distributions where direct sampling is challenging. We start with an example that demonstrates how to sample from a gamma distribution using this algorithm.

### Implementation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma

np.random.seed(123)
def MH_Gamma(n, a, b):
    mu = a / b
    sig = np.sqrt(a) / b
    vec = np.zeros(n)
    vec[0] = 3 * a / b
    for i in range(1, n):
        x = vec[i-1]  # current state
        x0 = np.random.normal(mu, sig)  # propose a new state
        
        # calculate the acceptance probability
        p_x0 = gamma.pdf(x0, a, scale=1/b)  # Target density at x0
        p_x = gamma.pdf(x, a, scale=1/b)  # Target density at x
        q_xx0 = norm.pdf(x0, mu, sig)  # proposal density at x given x0
        q_x0x = norm.pdf(x, mu, sig)  # proposal density at x0 given x
        
        aprob = min(1, (p_x0/p_x)*(q_xx0/q_x0x))  # Acceptance probability
        
        u = np.random.uniform(0, 1)  # Draw from uniform distribution
        
        if u < aprob:
            vec[i] = x0  # Accept the new state
        else:
            vec[i] = x  # Reject the new state and keep the old one
            
    return vec
```

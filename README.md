# Enhanced-Gibbs-Sampling-with-Metropolis-Hastings-Steps
This project explores the application of Bayesian inference techniques through the implementation of the Metropolis-Hastings algorithm and enhanced Gibbs sampling. The goal is to demonstrate how these methods can be used to sample from complex probability distributions and to perform Bayesian analysis in hierarchical models

The project is divided into two main parts:

** 1. Sampling from a Gamma Distribution Using Metropolis-Hastings:**
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
### Visualization
After performing the Metropolis-Hastings sampling, visualize the chain and the distribution of the sampled values.

```python
# Visulalization:
    # set parameters
nrep=55000
burnin=5000
shape=2.3
rate=2.7
vec=MH_Gamma(nrep, shape, rate)
len(vec)

#Modify the plots below so they apply only to the chain AFTER the burn-in period
vec_burnin=vec[burnin:]
len(vec_burnin)

mean_vec=np.mean(vec_burnin)

# create a subplot 2*1
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))

# Plot 1: Time Series Plot
ax1.plot(vec_burnin,label="Chain",color='blue')
ax1.axhline(y=mean_vec,color="red",linewidth=2,label=f"Mean:{mean_vec:.2f}")
ax1.set_xlabel("Chain")
ax1.set_ylabel('Draws')
ax1.set_title('Time Series of Draws')
ax1.legend()


# plot 2: Histogram
ax2.hist(vec_burnin,bins=30,density=True,color='gray', alpha=0.7,label="Simulated Density") 
ax2.axvline(x=mean_vec,color='red',linewidth=2,label=f'Mean:{mean_vec:.2f}')
ax2.set_xlabel("value")
ax2.set_ylabel('Density')
ax2.set_title("Histogram of Draws")

plt.tight_layout()
plt.show()
```
![Posterior of Theta_1](Time%20Series%20and%20Histogram%20of%20Draws.png)

** 2. Enhanced Gibbs Sampling with Metropolis-Hastings Steps for Hierarchical Models:**
Building on the foundational knowledge of MCMC techniques, the project extends to a more complex application involving enhanced Gibbs sampling for a hierarchical model. This approach combines direct sampling methods (Gibbs sampling) with Metropolis-Hastings steps to efficiently sample from the joint posterior distribution of parameters and hyperparameters within the model. The hierarchical model chosen for this application involves parameters $\theta_j$ and hyperparameters $(\alpha, \beta)$ illustrating the process of Bayesian inference in models where direct sampling of all parameters is not feasible. This part of the project emphasizes the calculation of posterior probabilities, updates of hyperparameters using Metropolis-Hastings steps, and the analysis of the sampled distributions through visualization techniques.

### Initialization and Data Preparation
First, we set up our observed data and initial parameter values:
```python
import numpy as np
from scipy.stats import gamma, norm, gammaln

# Observed data: counts of bicycles (y) and total vehicles (n) for streets with bike lanes
y = np.array([16, 9, 10, 13, 19, 20, 18, 17, 35, 55])
n = np.array([58, 90, 48, 57, 103, 57, 86, 112, 273, 64])

# Initialize hyperparameters
alpha, beta = 1.0, 1.0
n_iterations = 1000

```
### Storing Samples
Prepare arrays to store samples of $\theta_j$ and $(\alpha, \beta)$ for posterior analysis:
```python
samples_theta = np.zeros((n_iterations, len(y)))
samples_alpha = np.zeros(n_iterations)
samples_beta = np.zeros(n_iterations)

```
### Posterior Calculation Function
Calculate the posterior probability of $(\alpha, \beta)$  given the observed data $y$:
```python
def posterior_alpha_beta(alpha, beta, y):
    log_posterior = -2.5 * np.log(alpha + beta)
    for y_j in y:
        term = gammaln(alpha + y_j) - gammaln(alpha) + alpha * np.log(beta) - (alpha + y_j) * np.log(beta + 1)
        log_posterior += term
    return log_posterior

```

### Gibbs Sampling with Metropolis-Hastings Updates
Implement Gibbs sampling loop with Metropolis-Hastings updates for $(\alpha, \beta)$:
```python
for i in range(n_iterations):
    # Sampling for theta_j
    for j in range(len(y)):
        samples_theta[i, j] = gamma.rvs(a=alpha + y[j], scale=1/(beta + 1))
    
    # Update alpha using Metropolis-Hastings
    alpha_proposal = norm.rvs(loc=alpha, scale=0.1)
    if np.log(np.random.rand()) < posterior_alpha_beta(alpha_proposal, beta, y) - posterior_alpha_beta(alpha, beta, y):
        alpha = alpha_proposal
    
    # Update beta using Metropolis-Hastings
    beta_proposal = norm.rvs(loc=beta, scale=0.1)
    if np.log(np.random.rand()) < posterior_alpha_beta(alpha, beta_proposal, y) - posterior_alpha_beta(alpha, beta, y):
        beta = beta_proposal
    
    samples_alpha[i] = alpha
    samples_beta[i] = beta
```

### Visualization of Sampled Distributions
Plot histograms to visualize the posterior distributions of $\theta_1$, and $(\alpha, \beta)$:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(samples_theta[:, 0], bins=50, alpha=0.7)
plt.title('Posterior of $\\theta_1$')
plt.subplot(1, 3, 2)
plt.hist(samples_alpha, bins=50, alpha=0.7)
plt.title('Posterior of $\\alpha$')
plt.subplot(1, 3, 3)
plt.hist(samples_beta, bins=50, alpha=0.7)
plt.title('Posterior of $\\beta$')
plt.tight_layout()
plt.show()

```
![Posterior of Theta_1](Histographs%20of%20the%20draws%20of%20parameter%20and%20hyperparamters.png)



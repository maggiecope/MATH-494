"""
File: Assignment3.py 
Author: Maggie Cope 
Last Modified: 22 February 2024 
Description: Program that generates and plots different distributions of the sum of M PRNS 
""" 

import numpy as np
from numpy import random 
import matplotlib.pyplot as plt

# Generate M 
M = (4+8+6+1) # 4861 = the last 4 digits of my USD ID #
print(M)

# Define N (number of iterations)
N = 100000

# Generate M PRNs from U(0,1) distribution and find their sum, repeat N times 
uniform_samples = []
for i in range(N): 
    r = random.rand(M)
    sum = np.sum(r)
    uniform_samples.append(sum)

# Plot histogram for uniform distribution 
plt.hist(uniform_samples, bins=50, histtype='step', label='Uniform Distribution')

# Generate N samples from the normal distribution with appropriate parameters(mu and sigma)
mu = M / 2  # M/2 is the mean of the sum of M random numbers from U(0, 1)
sigma = np.sqrt(M / 12)  # sqrt(M/12) is the standard deviation of the sum of M random numbers from U(0, 1) 
normal_samples = np.random.normal(mu, sigma, N)

# Plot histogram for normal distribution 
plt.hist(normal_samples, bins=50, histtype='step', label='Normal Distribution')

# Add labels and legend
plt.xlabel('Sum')
plt.ylabel('Frequency')
plt.title('Comparison of Uniform and Normal Distributions')
plt.legend()

# Show plot
plt.show()
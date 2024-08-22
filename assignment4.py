"""
File: assignment4.py 
Author: Maggie Cope 
Last Modified: 5 March 2024 
Description: Program that generates a small synthetic dataset to imitate/simulate real-life datasets.
""" 


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


A = 1 #an integer whose value is the last digit of your USD ID#
B = 6 #the next-to-last digit.

# Generate dataset
np.random.seed(42)
data = np.zeros((100, 5)) #Your dataset should have 100 rows (observations) and 5 columns (features, dimensions)

# Generate first two columns
data[:, 0] = np.random.uniform(A, 2*A, size=100)
data[:, 1] = np.random.uniform(B, 2*B, size=100)

# Generate Gaussian noise terms
noise_3 = np.random.normal(0, np.mean(data[:, :2]) * 0.1, size=100)
noise_4 = np.random.normal(0, np.mean(data[:, :2]) * 0.5, size=100)
noise_5 = np.random.normal(0, np.mean(data[:, :2]), size=100)

# Generate next three columns with linear combinations and noise
data[:, 2] = data[:, 0] + data[:, 1] + noise_3
data[:, 3] = data[:, 0] + 2*data[:, 1] + noise_4
data[:, 4] = 2*data[:, 0] + data[:, 1] + noise_5

#Display the three-dimensional cloud of points (use the third, fourth, and fifth columns as dimensions).fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 2], data[:, 3], data[:, 4])
ax.set_xlabel('Column 3')
ax.set_ylabel('Column 4')
ax.set_zlabel('Column 5')
plt.title('Three-dimensional cloud of points')
plt.show()

# Compute and display mean and standard deviation of each column
print("Mean and standard deviation of each column:")
for i in range(5):
    mean = np.mean(data[:, i])
    std_dev = np.std(data[:, i])
    print(f"Column {i+1}: Mean = {mean}, Standard Deviation = {std_dev}")

#Center and standardize all data (all columns).
centered_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Compute and display mean and standard deviation after centering and standardization
print("\nMean and standard deviation of each column after centering and standardization:")
for i in range(5):
    mean = np.mean(centered_data[:, i])
    std_dev = np.std(centered_data[:, i])
    print(f"Column {i+1}: Mean = {mean}, Standard Deviation = {std_dev}")

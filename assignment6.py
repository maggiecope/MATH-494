"""
Math 494: Assignment 6 
Maggie Cope 
20 March 2024
Creates a noisy dataset based on a linear combination pattern, and uses multiple regression to discover the pattern. 
Compare the results of regression from scratch with the results obtained from the sklearn LinearRegression procedure.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Generate a dataset of 100 rows and 4 columns, all from the U(0, 1) distribution.
np.random.seed(0)
data = np.random.rand(100, 4)

# The coefficients are the four last digits of your USD ID#
coefficients = [4, 8, 6, 1]

# compute the linear combination of the first four columns
output = np.dot(data, coefficients)

# Add noise to the fifth column
noise_std = np.mean(coefficients) / 5
noise = np.random.normal(0, noise_std, 100)
output += noise

# Code the multiple regression using the normal equation of regression
# Normal equation of regression: X = (AtA)^-1(AtB)

# Add bias term 
B = np.c_[np.ones((data.shape[0], 1)), data]
# Compute normal equation
A_T = B.T  # At
A_TA_inverse = np.linalg.inv(np.dot(A_T, B))  # (AtA)^-1
A_T_B = np.dot(A_T, output)
X_B = np.dot(A_TA_inverse, A_T_B)  # X = (AtA)^-1(AtB)
X_normal = X_B[1:] # ignore bias term


# Display the resulting coefficients, which should be close to the “hidden” pattern.
print("Coefficients: ", coefficients)
print("Normal Equation Coefficients:", X_normal)

# Run the sklearn LinearRegression procedure on the very same dataset
model = LinearRegression()
model.fit(data, output)
X_sklearn = model.coef_

# Compare the results with your own regression
print("Sklearn Coefficients:", X_sklearn)

# Display (on the screen) a conclusion from the comparison: how close the results are to the actual pattern and to each other.

# Relative Error = Absolute Error/True Value
relative_error_normal = np.abs((X_normal - coefficients ))/ coefficients
print("Relative Error (Normal Equation):", relative_error_normal)

relative_error_sklearn = np.abs((X_sklearn - coefficients ))/ coefficients 
print("Relative Error (Sklearn):", relative_error_sklearn)

dif_normal_sklearn = np.abs(X_normal-X_sklearn)/X_sklearn 
print("Difference between Normal and Sklearn: ", dif_normal_sklearn)
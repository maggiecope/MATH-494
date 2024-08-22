"""""
Assignment #7
Maggie Cope
April 9, 2024 

Description:  You will create a noisy dataset based on a linear combination pattern, 
and you will use singular value decomposition (SVD) to reduce dimensionality of the dataset.
"""""
import numpy as np 

# Create a matrix M x 5 (you may use M = 20 or a similar value). 
# Fill the first two columns with uniformly distributed random numbers in the range from 0.0 to 10.0. 
M = 20 
matrix = np.random.uniform(0.0, 10.0, (M, 5))

# Display the matrix.
print("Original Matrix:")
print(matrix)

#The next three columns will be linear combinations of the first two. 
#The third column will be the sum of the first two. 
matrix[:,2] = matrix[:,0] + matrix[:,1]
#The fourth column will be ‘A’ times the first column plus ‘B’ times the second one, 
#where ‘A’, and ‘B’ are the last and next-to-last digits of your USD ID #
A = 2
B = 4
matrix[:,3]= A * matrix[:,0] + B * matrix[:,1]
#The fifth column will be ‘B’ times the last plus ‘A’ times next-to-last digit. 
matrix[:,4] = B * matrix[:,3] + A * matrix[:,2]

#Add noise to the third, fourth, and fifth column: add a Gaussian with the mean of 0.0 and standard deviation of about 1.0 - 2.0. 
noise_std = 1.5
matrix[:, 2:] += np.random.normal(0.0, noise_std, (M, 3))

#Perform the SVD decomposition and display matrices U, S, and VT.
X = matrix
#Compute the covariance matrix
cov_matrix = np.dot(X.T, X)
#Compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
#Compute the singular values
singular_values = np.sqrt(eigenvalues)
#Compute the left singular vectors
U = np.dot(X, eigenvectors) / singular_values
#Compute the right singular vectors
VT = eigenvectors.T

#display matrices
print("\nStep-by-Step SVD Decomposition:")
print("U matrix:")
print(U)
print("\nS matrix:")
print(np.diag(singular_values))
print("\nVT matrix:")
print(VT)

# Compare with np.linalg.svd()
U_np, S_np, VT_np = np.linalg.svd(X)

print("\nComparison with np.linalg.svd():")
print("U matrix:")
print(U_np)
print("\nS matrix:")
print(np.diag(S_np))
print("\nVT matrix:")
print(VT_np)

# Ask the user for the value of 'k'
k = int(input("Enter the value of 'k' (0, 1, 2, or 3): "))

# Reduce the dimensionality by 'k'
reduced_S = np.zeros((5, 5))
reduced_S[:5-k, :5-k] = np.diag(singular_values[:5-k])

#Display the reduced matrix. 
print("\nReduced S matrix:")
print(reduced_S)

#Compute and display the average relative error
# Compute the reduced version of matrix X
reduced_X = U @ reduced_S @ VT

# Compute the average relative error
relative_errors = np.abs((matrix - reduced_X) / matrix)
avg_relative_error = np.mean(relative_errors)

#Display average relative error
print("\nAverage Relative Error:")
print(avg_relative_error)
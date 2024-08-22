"""
File: assignment2.py 
Author: Maggie Cope 
Last Modified: 19 February 2024 
Description: Python program that illustrates, step-by-step, the process of eigendecomposition.
The results of each step of the computations should be clearly displayed on the screen.
"""

import numpy as np

# Begin with the matrix A
A = np.array([[1,2,3],[3,1,2]])

# Print the orginal Matrix A 
print("\nMatrix A: ")
print(A)


# Case 1: A^T * A 
print("\nCase 1: Matrix A^T * A  ")

# Compute the product of the transpose of A and A
AtA = np.matmul(A.T, A)

# Print the result of A^T * A
print("\nMatrix A^T * A: ")
print(AtA)

# Compute eigenvalues and eigenvectors of A^T * A
eigen_values_AtA, eigen_vectors_AtA = np.linalg.eig(AtA)

#print the eigenvalues and eigenvectors 
print("\n Eigenvalues and Eigen Vectors of A^T * A: ")
print("\n Eigenvalues: ", eigen_values_AtA)
print("\n Eigenvectors: ", eigen_vectors_AtA)

# Assign eigenvectors to matrix P for A^T * A
p_AtA = eigen_vectors_AtA
print("\n Matrix P for A^T * A : ", p_AtA)

# Create diagonal matrix D with eigenvalues of A^T * A
d_AtA = np.array([[0.0]*len(eigen_values_AtA)] * len(eigen_values_AtA))
for i in range(len(eigen_values_AtA)):
    d_AtA[i, i] = eigen_values_AtA[i]
print("\n Matrix D for A^T * A: ", d_AtA)

# Compute the inverse of matrix P for A^T * A
inverse_p_AtA = np.linalg.inv(p_AtA) 
print("\n Matrix P^-1 for A^T * A : ", inverse_p_AtA)

# Confirm that the product of these three matrices is equal to the original matrix
AtA_product = np.matmul(np.matmul(p_AtA, d_AtA), inverse_p_AtA)
print("\n AtA product")
print(AtA_product)


# Case 2: A * A^T
print("\nCase 2: A * A^T")

# Compute the product of A and its transpose
AAt = np.matmul(A, A.T)

# Print the result of A * A^T
print("\nMatrix A * A^T: ")
print(AAt)

# Compute eigenvalues and eigenvectors of A * A^T
eigen_values_AAt, eigen_vectors_AAt = np.linalg.eig(AAt)

# Print the eigenvalues and eigenvectors
print("\n Eigenvalues and Eigen Vectors of A * A^T : ")
print("\n Eigenvalues: ", eigen_values_AAt)
print("\n Eigenvectors: ", eigen_vectors_AAt)

# Assign eigenvectors to matrix P for A * A^T
p_AAt = eigen_vectors_AAt
print("\n Matrix P for A * A^T: ", p_AAt)

# Create diagonal matrix D with eigenvalues of A * A^T
d_AAt = np.array([[0.0]*len(eigen_values_AAt)] * len(eigen_values_AAt))
for i in range(len(eigen_values_AAt)):
    d_AAt[i, i] = eigen_values_AAt[i]
print("\n Matrix D for A * A^T: ", d_AAt)

# Compute the inverse of matrix P for A * A^T
inverse_p_AAt = np.linalg.inv(p_AAt) 
print("\n Matrix P^-1 for A * A^T: " , inverse_p_AAt)

# Confirm that the product of these three matrices is equal to the original matrix
AAt_product = np.matmul(np.matmul(p_AAt, d_AAt), inverse_p_AAt)
print("\n AAT product")
print(AAt_product)


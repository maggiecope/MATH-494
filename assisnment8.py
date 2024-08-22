import numpy as np 
from sklearn.decomposition import PCA

# Generate the same noisy dataset from Assignment #7.
M = 20 
matrix = np.random.uniform(0.0, 10.0, (M, 5)) 
matrix[:,2] = matrix[:,0] + matrix[:,1]
A = 1
B = 6
matrix[:,3]= A * matrix[:,0] + B * matrix[:,1]
matrix[:,4] = B * matrix[:,3] + A * matrix[:,2]
noise_std = 1.5
matrix[:, 2:] += np.random.normal(0.0, noise_std, (M, 3))

# Center the dataset
## Calculate the mean of each col  
means = np.mean(matrix, axis=0)
## Subtract the means from each col
centered_matrix = matrix - means 

# Perform PCA by hand
## Compute the covariance matrix 
cov_matrix = np.cov(centered_matrix, rowvar=False)
## Compute eigenvectors and eigenvalues 
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
## Sort eigenvalues and corresponding eigenvectors in descending order
sorted = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted]
eigenvectors_sorted = eigenvectors[:, sorted]
## Calculate total variance 
total_variance = np.sum(eigenvalues_sorted)

# Determine and display the amounts/proportions of variance of the data represented by each principal component.
## Calculate 
variance_proportions = eigenvalues_sorted / total_variance

## Display 
for i, variance in enumerate(variance_proportions, 1):
    print(f"Principal Component {i}: {variance:.2%} variance represented")


#  Perform PCA with the sklearn's PCA procedure.
pca = PCA()
pca.fit(matrix)
print("\nsklearn's PCA procedure: \n",pca.singular_values_ ,"\n")

# Perform SVD using numpy's svd function
U, s, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

# Display the singular values
print("numpy's SVD procedure: \n", s)



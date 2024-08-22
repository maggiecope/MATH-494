import numpy as np
import matplotlib.pyplot as plt 

# Use the last digit of your USD ID # for ‘a’ 
a = 1

# Ask the user to enter the value of ‘b’.
b = float(input("Enter a value for b: "))

# Generate a dataset of 100 observations
# where x is between 0 and 5
x = np.linspace(0, 5, 100)
# y is computed from the exponential formula.
y = a * np.exp(b * x)

# Add substantial Gaussian noise to each computed value of y 
# (meaning the exponential trend should be maintained but the pattern should be visibly noisy).
mu = 0 
sigma = 0.1 * max(y)
noise = np.random.normal(mu, sigma, len(x))
y_noisy = y + noise

# Display the noisy dataset (cloud of points).
plt.scatter(x, y_noisy, label='Noisy Data')
plt.plot(x, y, color='red', label='Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Noisy Dataset')
plt.legend()
plt.show()

# Linearize the dataset
valid_data = y_noisy > 0
x_valid = x[valid_data]
y_valid = y_noisy[valid_data]
y_linearized = np.log(y_valid)

# Perform linear regression on the filtered data
sum_x = np.sum(x_valid)
sum_y = np.sum(y_linearized)
sum_xx = np.sum(x_valid**2)
sum_xy = np.sum(x_valid * y_linearized)

# Calculate estimates for a and b using formulas from class 
n = len(x_valid)  # number of valid observations
a_estimate = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
b_estimate = (sum_y - a_estimate * sum_x) / n

# Printing estimated values of a and b
print("Estimated value of a:", a_estimate)
print("Estimated value of b:", b_estimate)

# Comparison with original values
print("Original value of a:", a)
print("Original value of b:", b)

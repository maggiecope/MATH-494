# assignment1.py 
# Math 494 Assignment #1
# Author: Maggie Cope
# Date: 2/11/24
#
# Writes a script that uses gradient descent to find the minimum of the function f(x,y,z) = x^2+y^2+z^2-x-y+z
# Displays a curve that illustrates how the change of the values of (x, y, z) 
# between two consecutive iterations changes with respect to the iteration number.
# Determines the largest value of step size (eta - learning rate) for which the method converges.

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Define the function f(x,y,z) = x^2+y^2+z^2-x-y+z
def f(x, y, z): 
    return x**2 + y**2 + z**2 - x - y + z

# Partial derivatives of f 
def fx(x, y, z): 
    return 2*x - 1 

def fy(x, y, z):
    return 2*y - 1 

def fz(x, y, z): 
    return 2*z + 1

# (Hyper-)parameters of gradient descent:
eta = 0.001         # learning rate
eps = 1e-6          # minimum step size
max_cnt = 10000     # max number of iterations

# Starting point
x = 0.0
y = 0.0
z = 0.0

# Lists to store changes and iteration number
changes = []
iterations = []

change = 1          # an artificial starting value


# The gradient descent loop
cnt = 0
while (cnt < max_cnt) and (change > eps):
    x_change = - eta * fx(x, y, z)
    y_change = -eta * fy(x, y, z)
    z_change = -eta * fz(x, y, z)
    x += x_change
    y += y_change
    z += z_change
    # see how big a step we made
    change = norm(np.array([x_change, y_change, z_change]))
    
    # Store changes and iteration numbers
    changes.append(change)
    iterations.append(cnt)

    cnt += 1

# Plot change curve
plt.plot(iterations, changes)  # Start from index 1 to skip the first iteration
plt.xlabel('Iteration')
plt.ylabel('Change in (x, y, z)')
plt.title('Change in (x, y, z) between consecutive iterations')
plt.show()

# Gradient descent loop to determine largest value of step size 
max_eta =0.0
for eta in np.linspace(0.001, 1.0, 10000):
    x = 0.0
    y = 0.0
    z = 0.0
    cnt = 0
    while (cnt < max_cnt):
        x_change = x - eta * fx(x, y, z)
        y_change = y - eta * fy(x, y, z)
        z_change = z - eta * fz(x, y, z)
        
        # Calculate change in parameters 
        change = np.linalg.norm([x_change - x, y_change - y, z_change - z])
        cnt += 1

        x = x_change
        y = y_change
        z = z_change

        if change < eps:
            if eta > max_eta:
                max_eta = eta
            break 

print("Largest converging step size (eta):", max_eta)
import numpy as np
import matplotlib.pyplot as plt

def stable_softmax(x):
    """
    Computes the softmax function in a numerically stable way.
    
    Parameters:
        x (numpy array): Input vector.
    
    Returns:
        numpy array: Softmax probability vector.
    """
    m = np.max(x)  # Get the maximum value of the input vector
    exp_x = np.exp(x - m)  # Subtract the maximum before computing the exponential
    return exp_x / np.sum(exp_x)

# Example usage
x = np.linspace(-10, 10, 100)
softmax_values = stable_softmax(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, softmax_values, label="Stable Softmax", color='b')
plt.xlabel("x")
plt.ylabel("Softmax(x)")
plt.title("Stable Softmax Function")
plt.legend()
plt.grid()
plt.show()

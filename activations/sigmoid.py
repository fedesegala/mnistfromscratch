import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function."""
    z = np.clip(z, -100, 100)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Compute the derivative of the sigmoid function."""
    sigmoid_output = sigmoid(z)
    return sigmoid_output * (1 - sigmoid_output)

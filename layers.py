import numpy as np

# neg to 0, pos same
def relu(X):
    return np.maximum(0, X)

# 1.0 when relu was unchanged, 0.0 if it was changed
def relu_deriv(x):
    return (x > 0).astype(float)

# raw scores to probabilities (add to 1)
# no overflow, so we subtract max 
def softmax(X):
    e = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

class Linear:

    # init weights and biases 
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    # dot product of X and weights plis the bias
    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, dZ):
        self.dZ = dZ
        self.dW = np.dot(self.X.T, self.dZ)  # gradient for weights
        self.db = np.sum(dZ, axis=0, keepdims=True)  # gradient for biases
        self.dX = np.dot(dZ, self.W.T)  # gradient to pass to previous layer
        return self.dX
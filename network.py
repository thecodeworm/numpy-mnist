import numpy as np
import layers as l
from layers import Linear

class MultiLayerPerceptron:

    # two layers: there are 784 pixels in an image, down to 128 features, and then 10 digits which are 0 - 9 
    def __init__(self):
        self.layer1 = Linear(784, 128)
        self.layer2 = Linear(128, 10)
    
    def forward(self, X):
        self.a1 = l.relu(self.layer1.forward(X))      # learn the features from the pixls given
        self.a2 = l.softmax(self.layer2.forward(self.a1))  # turn the features into probabilities
        return self.a2
    
    def backward(self, Y):
        self.dZ2 = np.subtract(self.a2, Y)             # how wrong were predictions
        self.dA1 = self.layer2.backward(self.dZ2)      # backprop error through layer2
        self.dZ1 = self.dA1 * l.relu_deriv(self.a1)   # block gradients where relu zeroed out
        return self.layer1.backward(self.dZ1)          # backprop error through layer1
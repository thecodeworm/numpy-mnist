# MNIST Classifier from Scratch

A handwritten digit classifier built using only NumPy. No PyTorch, no TensorFlow, no ML libraries of any kind. Every component including the forward pass, backpropagation, and weight updates are implemented manually using matrix operations.

Achieves 98% accuracy on the MNIST test set.


## Architecture

Multi-layer perceptron with one hidden layer:

Input (784) -> Hidden (128, ReLU) -> Output (10, Softmax)

Each 28x28 pixel image is flattened into a 784-dimensional vector. This vector is passed through a linear transformation (Z = X @ W + b) into 128 hidden neurons. ReLU is applied as the activation function, zeroing out negative values and introducing non-linearity. Without a non-linear activation, the entire network would collapse into a single linear transformation regardless of depth.

The hidden layer output is passed through a second linear transformation into 10 output neurons, one per digit class. Softmax converts these raw scores into a probability distribution summing to 1. The predicted digit is the class with the highest probability.

Loss is computed using cross-entropy, which penalizes confident wrong predictions heavily. Gradients are computed via backpropagation using the chain rule and flow backwards through softmax, layer 2, ReLU, and layer 1. At each layer dW and db are computed and weights are updated via gradient descent.


## Files

data.py       - downloads and parses MNIST IDX binary files, normalizes pixels to [0, 1], flattens images to 784-dim vectors, one-hot encodes labels, splits into train/val/test
layers.py     - Linear class with forward and backward methods, ReLU, ReLU derivative, and Softmax
network.py    - MLP class wiring both layers with full forward and backward passes
train.py      - training loop with mini-batch gradient descent, prints loss and validation accuracy each epoch, saves weights to disk
evaluate.py   - loads saved weights into a fresh model and reports final test accuracy
demo.ipynb    - interactive notebook, click a button to run a random test image through the model and see the prediction and confidence


## Usage

1. Download the 4 MNIST IDX files and place them in data/
   - train-images-idx3-ubyte
   - train-labels-idx1-ubyte
   - t10k-images-idx3-ubyte
   - t10k-labels-idx1-ubyte
2. Run python train.py to train and save weights to weights/
3. Run python evaluate.py to check final test accuracy
4. Open demo.ipynb and click the button to test live predictions


## Training Details

Loss function:  cross-entropy
Optimizer:      mini-batch gradient descent
Batch size:     32
Epochs:         20
Learning rate:  0.01
Train set:      50,000 samples
Val set:        10,000 samples
Test set:       10,000 samples


## Weight Initialization

Weights are initialized using np.random.randn scaled by 0.01 to keep values small at the start of training. Large initial weights cause the softmax output to saturate, making gradients vanish and slowing convergence. Biases are initialized to zero.


## Requirements

numpy
matplotlib
jupyter
ipywidgets
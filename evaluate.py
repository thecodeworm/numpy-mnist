import numpy as np
from network import MultiLayerPerceptron as MLP
from data import X_test, Y_test

"""
From train.py:

np.save('weights/layer1W.npy', model.layer1.W)
np.save('weights/layer1b.npy', model.layer1.b)
np.save('weights/layer2W.npy', model.layer2.W)
np.save('weights/layer2b.npy', model.layer2.b)

"""
model = MLP()
model.layer1.W = np.load('weights/layer1W.npy')
model.layer1.b = np.load('weights/layer1b.npy')
model.layer2.W = np.load('weights/layer2W.npy')
model.layer2.b = np.load('weights/layer2b.npy')

test_preds = np.argmax(model.forward(X_test), axis=1)
test_true = np.argmax(Y_test, axis=1)
test_acc = np.mean(test_preds == test_true)  
print(f"In 50 epochs, the model got a test acc. of {test_acc:4f}")
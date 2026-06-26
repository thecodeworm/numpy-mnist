import numpy as np
from data import X_train, Y_train, X_val, Y_val
from network import MultiLayerPerceptron as MLP


model = MLP()
epochs = 20
learning_rate = 0.01

for epoch in range(epochs):
    for i in range(0, len(X_train), 32):
        X_batch = X_train[i:i+32]
        Y_batch = Y_train[i:i+32]

        Y_hat = model.forward(X_batch)
        loss = -np.sum(Y_batch * np.log(Y_hat))/len(X_batch)
        model.backward(Y_batch)

        model.layer1.W -= learning_rate * model.layer1.dW
        model.layer1.b -= learning_rate * model.layer1.db
        model.layer2.W -= learning_rate * model.layer2.dW
        model.layer2.b -= learning_rate * model.layer2.db
    val_preds = np.argmax(model.forward(X_val), axis=1)
    val_true = np.argmax(Y_val, axis=1)
    val_acc = np.mean(val_preds == val_true)  
    print(f"Epoch {epoch+1}/{epochs}\nLoss: {loss:.4f}\nVal Acc: {val_acc:.4f}")


np.save('weights/layer1W.npy', model.layer1.W)
np.save('weights/layer1b.npy', model.layer1.b)
np.save('weights/layer2W.npy', model.layer2.W)
np.save('weights/layer2b.npy', model.layer2.b)

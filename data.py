import numpy as np

def load_data(image_path, label_path):
    with open(image_path, 'rb') as file:
        file.read(16)
        image_data = file.read()
        image_arr = np.frombuffer(image_data, dtype=np.uint8)
        
    with open(label_path, 'rb') as file:
        file.read(8)
        label_data = file.read()
        label_arr = np.frombuffer(label_data, dtype=np.uint8)

    X = image_arr.reshape(-1, 784)
    X = X/255.0
    N = len(label_arr)

    Y = np.zeros((N, 10))
    Y[np.arange(N), label_arr] = 1

    return X, Y


X_train_full, Y_train_full = load_data('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte')
X_test, Y_test = load_data('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')

# last 10k val
X_train, Y_train = X_train_full[:50000], Y_train_full[:50000]
X_val, Y_val = X_train_full[50000:], Y_train_full[50000:]
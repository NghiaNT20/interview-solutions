import numpy as np
import struct
import os
import gzip
import pickle
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def read_idx(filename):
    with gzip.open(filename, "rb") as f:
        zero, data_type, dims = struct.unpack(">HBB", f.read(4))
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(path):
    train_images = read_idx(os.path.join(path, "train-images-idx3-ubyte.gz"))
    train_labels = read_idx(os.path.join(path, "train-labels-idx1-ubyte.gz"))
    test_images = read_idx(os.path.join(path, "t10k-images-idx3-ubyte.gz"))
    test_labels = read_idx(os.path.join(path, "t10k-labels-idx1-ubyte.gz"))
    return (train_images, train_labels), (test_images, test_labels)


data_path = "/root/personal/dunglq12/practice-ds/interview_exercise/datasets"
(X_train, y_train), (X_test, y_test) = load_mnist(data_path)

X_train = X_train.reshape(-1, 28 * 28).astype(float)
X_test = X_test.reshape(-1, 28 * 28).astype(float)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

lb = LabelBinarizer()
y_train_onehot = lb.fit_transform(y_train)
y_val_onehot = lb.transform(y_val)
y_test_onehot = lb.transform(y_test)


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2


with open("exercise_3/simple_nn_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Measure inference speed
start_time = time.time()
predictions = loaded_model.forward(X_test)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time for {len(X_test)} samples: {inference_time} seconds")
print(f"Average inference time per sample: {inference_time / len(X_test)} seconds")

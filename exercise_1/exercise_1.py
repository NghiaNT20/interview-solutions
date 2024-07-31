"""Bài 1: ML cơ bản (3đ)
        Xây dựng một mô hình Machine learning (not deep learning) ứng dụng cho 
        bài phân biệt loại ký tự quang học, ứng dụng data MNIST. Chỉ sử dụng numpy"""

import numpy as np
import gzip
import struct
import os


def read_idx(filename):
    with gzip.open(filename, "rb") as f:
        magic_number, num_items = struct.unpack(">II", f.read(8))
        if magic_number == 2051:  
            num_rows, num_cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_items, num_rows, num_cols
            )
        elif magic_number == 2049: 
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items)
        else:
            raise ValueError("Invalid magic number in file: {}".format(magic_number))
        return data


class LinearSVM:
    def __init__(self, learning_rate=0.01, num_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.C = C

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.W = np.zeros(num_features)
        self.b = 0

        for _ in range(self.num_iterations):
            for i in range(num_samples):
                if y[i] * (np.dot(X[i], self.W) + self.b) < 1:
                    self.W += self.learning_rate * (
                        y[i] * X[i] - 2 * 1 / self.C * self.W
                    )
                    self.b += self.learning_rate * y[i]
                else:
                    self.W -= self.learning_rate * 2 * 1 / self.C * self.W

    def predict(self, X):
        return np.sign(np.dot(X, self.W) + self.b)

    def save_model(self, filename):
        np.savez(filename, W=self.W, b=self.b)

    @staticmethod
    def load_model(filename):
        data = np.load(filename)
        model = LinearSVM()
        model.W = data["W"]
        model.b = data["b"]
        return model


path = "/root/personal/dunglq12/practice-ds/interview_exercise/datasets"

train_images = read_idx(os.path.join(path, "train-images-idx3-ubyte.gz"))
train_labels = read_idx(os.path.join(path, "train-labels-idx1-ubyte.gz"))
test_images = read_idx(os.path.join(path, "t10k-images-idx3-ubyte.gz"))
test_labels = read_idx(os.path.join(path, "t10k-labels-idx1-ubyte.gz"))

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

train_labels_binary = (train_labels == 0).astype(int) * 2 - 1
test_labels_binary = (test_labels == 0).astype(int) * 2 - 1


svm_model = LinearSVM(learning_rate=0.001, num_iterations=1000, C=1.0)
svm_model.fit(train_images, train_labels_binary)


svm_model.save_model("svm_model.npz")
loaded_model = LinearSVM.load_model("svm_model.npz")

svm_predictions = loaded_model.predict(test_images)
accuracy = np.mean(svm_predictions == test_labels_binary)
print(f"SVM Accuracy: {accuracy}")

import os
import gzip
import numpy as np
import struct

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        magic_number, num_items = struct.unpack(">II", f.read(8))
        if magic_number == 2051: 
            num_rows, num_cols = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        elif magic_number == 2049:  
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items)
        else:
            raise ValueError("Invalid magic number in file: {}".format(magic_number))
        return data


path = '/root/personal/dunglq12/practice-ds/interview_exercise/datasets'  


train_images = read_idx(os.path.join(path, 'train-images-idx3-ubyte.gz'))
train_labels = read_idx(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
test_images = read_idx(os.path.join(path, 't10k-images-idx3-ubyte.gz'))
test_labels = read_idx(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0


train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

def create_triplets(images, labels, num_triplets):
    triplets = []
    num_classes = np.max(labels) + 1
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for _ in range(num_triplets):
        anchor_class = np.random.randint(0, num_classes)
        negative_class = (anchor_class + np.random.randint(1, num_classes)) % num_classes
        
        anchor_idx = np.random.choice(digit_indices[anchor_class])
        positive_idx = np.random.choice(digit_indices[anchor_class])
        negative_idx = np.random.choice(digit_indices[negative_class])
        
        triplets.append((images[anchor_idx], images[positive_idx], images[negative_idx]))
    
    return np.array(triplets)

num_triplets = 10000
train_triplets = create_triplets(train_images, train_labels, num_triplets)

def initialize_weights(input_shape, hidden_units, output_units):
    weights = {
        'W1': np.random.randn(np.prod(input_shape), hidden_units) * 0.01,
        'b1': np.zeros((1, hidden_units)),
        'W2': np.random.randn(hidden_units, output_units) * 0.01,
        'b2': np.zeros((1, output_units))
    }
    return weights

def forward_propagation(X, weights):
    X_flat = X.reshape(X.shape[0], -1)  
    Z1 = np.dot(X_flat, weights['W1']) + weights['b1']
    A1 = np.maximum(0, Z1)  
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    return Z2

input_shape = train_images[0].shape
hidden_units = 128
output_units = 64
weights = initialize_weights(input_shape, hidden_units, output_units)

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.sum(np.square(anchor - positive), axis=-1)
    neg_dist = np.sum(np.square(anchor - negative), axis=-1)
    loss = np.maximum(pos_dist - neg_dist + alpha, 0)
    return np.mean(loss)

def compute_gradients(anchor, positive, negative, weights, alpha=0.2):
    anchor_embedding = forward_propagation(anchor[np.newaxis], weights)
    positive_embedding = forward_propagation(positive[np.newaxis], weights)
    negative_embedding = forward_propagation(negative[np.newaxis], weights)
    
    pos_dist = np.sum(np.square(anchor_embedding - positive_embedding), axis=-1)
    neg_dist = np.sum(np.square(anchor_embedding - negative_embedding), axis=-1)
    loss = np.maximum(pos_dist - neg_dist + alpha, 0)
    
    grad_W1 = np.zeros_like(weights['W1'])
    grad_b1 = np.zeros_like(weights['b1'])
    grad_W2 = np.zeros_like(weights['W2'])
    grad_b2 = np.zeros_like(weights['b2'])
    
    return grad_W1, grad_b1, grad_W2, grad_b2

def update_weights(weights, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate):
    weights['W1'] -= learning_rate * grad_W1
    weights['b1'] -= learning_rate * grad_b1
    weights['W2'] -= learning_rate * grad_W2
    weights['b2'] -= learning_rate * grad_b2


def train_siamese_network(train_triplets, weights, learning_rate=0.001, epochs=10):
    for epoch in range(epochs):
        epoch_loss = 0
        for anchor, positive, negative in train_triplets:
            anchor_embedding = forward_propagation(anchor[np.newaxis], weights)
            positive_embedding = forward_propagation(positive[np.newaxis], weights)
            negative_embedding = forward_propagation(negative[np.newaxis], weights)
            
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            epoch_loss += loss
            
            # Tính gradient
            grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients(anchor, positive, negative, weights)
            
            # Cập nhật trọng số
            update_weights(weights, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_triplets)}")


train_siamese_network(train_triplets, weights)

def save_model(weights, filename):
    np.savez(filename, W1=weights['W1'], b1=weights['b1'], W2=weights['W2'], b2=weights['b2'])

def load_model(filename):
    data = np.load(filename)
    weights = {
        'W1': data['W1'],
        'b1': data['b1'],
        'W2': data['W2'],
        'b2': data['b2']
    }
    return weights

def evaluate_model(test_images, test_labels, weights):
    correct = 0
    total = len(test_labels)
    
    for i in range(total):
        anchor_embedding = forward_propagation(test_images[i][np.newaxis], weights)
        
        for j in range(total):
            if i != j:
                test_embedding = forward_propagation(test_images[j][np.newaxis], weights)
                dist = np.sum(np.square(anchor_embedding - test_embedding))
                if (test_labels[i] == test_labels[j] and dist < 0.5) or (test_labels[i] != test_labels[j] and dist >= 0.5):
                    correct += 1
    
    accuracy = correct / (total * (total - 1))
    print(f'Accuracy: {accuracy}')

input_shape = train_images[0].shape
hidden_units = 128
output_units = 64
weights = initialize_weights(input_shape, hidden_units, output_units)

train_siamese_network(train_triplets, weights)

save_model(weights, 'siamese_model.npz')

loaded_weights = load_model('siamese_model.npz')

evaluate_model(test_images, test_labels, loaded_weights)


'''Bài 2: 
b. Viết công thức toán, implement (code numpy) và giải thích khi Input triplet loss mở rộng
không chỉ là 1 mẫu thật và một mẫu giả nữa mà sẽ là 2 mẫu thật và 5 mẫu giả.
'''

import numpy as np

def extended_triplet_loss(anchor, positives, negatives, alpha=0.2):
    losses = []
    for positive in positives:
        pos_dist = np.linalg.norm(anchor - positive)
        neg_dists = [np.linalg.norm(anchor - negative) for negative in negatives]
        min_neg_dist = np.min(neg_dists)
        loss = np.maximum(0, pos_dist - min_neg_dist + alpha)
        losses.append(loss)
    return np.sum(losses)

#Example
anchor = np.array([1.0, 5.0])
positives = [np.array([1.1, 2.1]), np.array([1.2, 2.2])]
negatives = [np.array([3.0, 4.0]), np.array([3.1, 4.1]), np.array([3.2, 4.2]), np.array([3.3, 4.3]), np.array([3.4, 4.4])]
alpha = 0.1

loss = extended_triplet_loss(anchor, positives, negatives, alpha)
print(f"Extended Triplet Loss: {loss}")
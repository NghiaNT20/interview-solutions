'''Bài 2: Triplet loss (3đ)
a. Viết công thức toán, implement (code numpy) và giải thích về Triplet loss.
'''


import numpy as np

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = np.linalg.norm(anchor - positive)
    neg_dist = np.linalg.norm(anchor - negative)

    loss = np.maximum(0, pos_dist - neg_dist + alpha)
    return loss
#Example
anchor = np.array([1.0, 5.0])
positive = np.array([1.1, 2.1])
negative = np.array([3.0, 4.0])
alpha = 0.15

loss = triplet_loss(anchor, positive, negative, alpha)
print(f"Triplet Loss: {loss}")
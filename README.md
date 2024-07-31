# Interview-solutions

# Exercise 1:
    SVM (Support Vector Machine) model

    Precision : 0.9404
 
# Exercise 2:
a. Formula:
Triplet loss is defined as follows:

$$
L(a, p, n) = \max(0, d(a, p) - d(a, n) + \alpha) 
$$

While:

a : is anchor 
p : is positive 
n : is negative 
d(x, y) : is the distance between two points ( x ) and ( y ) (usually Euclidean distance is used)
$ \alpha $ : is margin (border distance)

Explain:

    Triplet loss tries to ensure that the distance between the anchor and the positive is at least one margin ( \alpha ) smaller than the distance between the anchor and the negative.
    Otherwise, it generates a loss value to adjust the model.

b. 
When expanding triplet loss with 2 real samples and 5 fake samples, the formula can be written as follows:
$$
L(a, p_1, p_2, n_1, n_2, n_3, n_4, n_5) = \max(0, d(a, p_1) - \min(d(a, n_i)) + \alpha) + \max(0, d(a, p_2) - \min(d(a, n_i)) + \alpha) 
$$
While:

$p_1, p_2$ : are two positive samples
$n_1, n_2, n_3, n_4, n_5$ : are two negative samples

Explain:

      In the extended case, we calculate the distance between the anchor and each positive, and then compare it with the smallest distance between the anchor and the negatives. 
      Sum these loss values ​​to get the final loss value.This helps the model learn that the real samples should be closer to each other than any fake samples.


# Exercise 3: Explain the approach and analyze the advantages and disadvantages of Deeplearning method compared to Machine Learning method
Approach:

    Deep Learning with Triplet Loss: I found that using triplet loss for classification problems is not reasonable (basically it can still be trained). 
    So I used siamese network. Siamese Network combined with Triplet Loss is a powerful method to learn embeddings in recognition and search problems.


Pros:

    Deep Learning:
    - Deep Learning is capable of learning complex features from data.
    - Deep Learning can learn more complex models than Machine Learning.
    - Deep Learning can learn models without having to define features in advance.
    
    Machine Learning:
    - Machine Learning can work well with small data.
    - Machine Learning can easily explain the learned model.
    - Machine Learning can work well with data that has clear features.

Cons:

    Deeplearning:
    - Deep Learning requires a large amount of data to train the model.
    - Deep Learning requires a large amount of data to train the model.
    - Deep Learning requires a large amount of data to train the model.
    
    Machine Learning:
    - Machine Learning cannot learn complex models like Deep Learning.
    - Machine Learning needs to identify features before training the model.
    - Machine Learning cannot learn complex models like Deep Learning.
    
Summary: 

    Compared with traditional Machine Learning methods such as k-NN, Deep Learning with Triplet Loss can learn more complex features from data, but requires more resources and training time.

# Exercise 4:

1. Build API with Flask
2. Package Application with Docker
## Run Dockerfile:
- docker build -t my_flask_app .
- docker run -it -p 8000:8000 --name my_flask_container -v $(pwd):/app -d my-flask-app /bin/bash

    Test API with Postman
3. Set up CI/CD with GitHub Actions
    Create a .github/workflows/ci_cd.yml file to configure CI/CD pipeline.
4. Automate Testing with Pytest
5. Automate Docker Build
    Docker will be automatically built in CI/CD pipeline when there is a new commit to the "master" branch.
6. Run CI/CD Pipeline

    

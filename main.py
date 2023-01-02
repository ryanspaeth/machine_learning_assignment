from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np

# Loading the data set
mninst = fetch_openml('mnist_784')
data = mninst.data.to_numpy()

dataset_images = np.reshape(data, (-1, 28, 28))

y = np.array(mninst.target)

# creating the matrix where the image features are extracted
feature_matrix = np.zeros((dataset_images.shape[0], 4))

# feature: getting the average pixel value per image
for index in range(len(dataset_images)):
    total = 0
    for y in dataset_images[index]:
        for x in y:
            total += x
    average = total / (28 * 28)
    feature_matrix[index][0] = average

# feature: getting black pixels coordinates
# threshold when a pixel is black
threshold_black = 5
for index in range(len(dataset_images)):
    counter = 0
    for y in dataset_images[index]:
        for x in y:
            if x <= threshold_black:
                counter = counter + 1
    feature_matrix[index][1] = counter

print(feature_matrix[0][1])

from sklearn.datasets import fetch_openml
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

# feature: symmetry x-axis
for index in range(len(dataset_images)):
    sum_top = 0
    sum_bot = 0
    for y in range(len(dataset_images[index])//2):
        for x in range(len(dataset_images[index][0])):
            sum_top += dataset_images[index][y][x]
            sum_bot += dataset_images[index][y + len(dataset_images[index])//2][x]
    feature_matrix[index][2] = sum_top - sum_bot

# feature: symmetry y-axis
for index in range(len(dataset_images)):
    sum_left = 0
    sum_right = 0
    for y in range(len(dataset_images[index])):
        for x in range(len(dataset_images[index][0])//2):
            sum_left += dataset_images[index][y][x]
            sum_right += dataset_images[index][y][x + len(dataset_images[index][0])//2]
    feature_matrix[index][3] = sum_left - sum_right


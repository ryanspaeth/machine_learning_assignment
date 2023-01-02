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

# b) calculate correlation between feature 1 and 2
feature1_list = []
feature2_list = []
for index in feature_matrix:
    feature1_list.append(index[0])
    feature2_list.append(index[1])

correlation_1_2 = np.corrcoef(feature1_list,feature2_list)
print(correlation_1_2[0][0])

"""
Result: -0.97 -> That means it is negatively correlated. The more black values which are in the image the 
the less the average value is. That makes sense, because in an black/white image black = 0 and white = 255.
Therefore, the more black values the more pixels have a value of 0 which means the average pixel value will
also be smaller.
Because they have a high correlation, you can remove one of the variables because using both of them increases
the dimensionality without adding extra information.
"""

# c)
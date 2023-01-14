"""
Author: Ryan Spaeth
"""
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

# Loading the data set
mninst = fetch_openml('mnist_784')
data = mninst.data.to_numpy()

dataset_images = np.reshape(data, (-1, 28, 28))

target = np.array(mninst.target)
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
threshold_black = 200

for index in range(len(dataset_images)):
    counter = 0
    for y in dataset_images[index]:
        for x in y:
            if x >= threshold_black:
                counter = counter + 1
    feature_matrix[index][1] = counter

# feature: symmetry x-axis
for index in range(len(dataset_images)):
    sum_top = 0
    sum_bot = 0
    for y in range(len(dataset_images[index]) // 2):
        for x in range(len(dataset_images[index][0])):
            sum_top += dataset_images[index][y][x]
            sum_bot += dataset_images[index][y + len(dataset_images[index]) // 2][x]
    feature_matrix[index][2] = sum_top - sum_bot

# feature: symmetry y-axis
for index in range(len(dataset_images)):
    sum_left = 0
    sum_right = 0
    for y in range(len(dataset_images[index])):
        for x in range(len(dataset_images[index][0]) // 2):
            sum_left += dataset_images[index][y][x]
            sum_right += dataset_images[index][y][x + len(dataset_images[index][0]) // 2]
    feature_matrix[index][3] = sum_left - sum_right

# b) calculate correlation between feature 1 and 2
feature1_list = []
feature2_list = []
for index in feature_matrix:
    feature1_list.append(index[0])
    feature2_list.append(index[1])

correlation_1_2 = np.corrcoef(feature1_list, feature2_list)
print(correlation_1_2[0][1])

"""
Result: 0.97 -> That means it is correlated. The more black values which are in the image the 
the higher the average value is. That makes sense, because in an black/white image black = 255 and white = 0.
Therefore, the more black values the more pixels have a value of 255 which means the average pixel value will
also be higher.
Because they have a high correlation, you can remove one of the variables because using both of them increases
the dimensionality without adding extra information.
"""

# c) Apply PCA to the data and create a scatterplot of it
pca = PCA()
feature_matrix = pca.fit_transform(feature_matrix, target)
plot = plt.scatter(x=feature_matrix[:50, 0], y=feature_matrix[:50, 2], c=target[:50].astype(np.int32))
plt.legend(handles=plot.legend_elements()[0])
plt.show()

# d) split training and test into 60 40
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target, test_size=0.4, random_state=0)

# 2.1 SVM
# a) Create a Linear SVM(soft margin)

# parameters for the grid search cross validation
parameters = {'C': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]}
# parameters = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
# linear_models = []

svm_linear = SVC(kernel='linear', max_iter=1000)
search = GridSearchCV(svm_linear, parameters, scoring='accuracy', n_jobs=1, cv=10, refit=True)
# execute search
result = search.fit(X_train, y_train)
print(result.best_estimator_)
print(result.best_params_)
"""It shows that the param of C: 0.001 has the best results."""

y_pred = result.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))

"""
 0.12 % accuracy becauwe max_iter 1000
 0.12 % f1_score because max_iter 1000 
"""
# b) Create a RBG Kernel SVM
# parameters for the grid search cross validation
parameters = {'C': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
              'gamma': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]}

# creating rgb SVM model and Grid search
svm_rgb = SVC(kernel='rbf', max_iter=10)
clf_rgb = GridSearchCV(svm_rgb, parameters, scoring='accuracy')
""" Best model: {'C': 0.01, 'gamma': 0.005}"""
# training the model
result = clf_rgb.fit(X_train, y_train)
print(result.best_params_)

y_pred = result.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))
"""
   Accuracy: 0.11360714285714285
   F1_Score: 0.11360714285714284, cos max_iter 10, f1_score with sklearn and macro returns accuracy ? source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
"""
# creating KNN model
parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]}
neigh = KNeighborsClassifier()
clf_knn = GridSearchCV(neigh, parameters, scoring='accuracy')
result = clf_knn.fit(X_train, y_train)
print(result.best_params_)
"""
    Best model: {'n_neighbors': 23}
"""

y_pred = result.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))
"""
    Accuracy: 0.35328571428571426
    F1-Score: 0.35328571428571426
"""

# creating naive bayes model
gnb = GaussianNB()
result = gnb.fit(X_train, y_train)


y_pred = result.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))

""" 
    Accuracy: 0.4551071428571429
    F1_Score: 0.4551071428571429
"""
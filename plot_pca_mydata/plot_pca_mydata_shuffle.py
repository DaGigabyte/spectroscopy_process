import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import model_selection
import glob
import csv
import os

PET_list = list()
for filename in os.listdir("PET"):
    filename = "PET\\" + filename
    with open(filename, 'r') as file:
        PET_list += list(csv.reader(file, delimiter=','))
PP_list = list()
for filename in os.listdir("PP"):
    filename = "PP\\" + filename
    with open(filename, 'r') as file:
        PP_list += list(csv.reader(file, delimiter=','))

arr1 = np.asarray(PET_list)
arr1 = np.delete(arr1, 10, 1)
arr1_offseted = arr1.copy()
# for ar in arr1_offseted:
#     ar -= ar[-1]
arr2 = np.asarray(PP_list)
arr2 = np.delete(arr2, 10, 1)
arr2_offseted = arr2.copy()
# for ar in arr2_offseted:
#     ar -= ar[-1]

X = np.vstack((arr1_offseted, arr2_offseted))
sample_pair = [("PET", 1), ("PP", 5)]
Y = np.hstack((np.repeat(1, arr1_offseted.shape[0]), np.repeat(5, arr2_offseted.shape[0])))

# Train test split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.5, random_state=42)

fig = plt.figure(1, figsize=(4, 3))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
pca = decomposition.PCA(n_components=3)
pca.fit(X_train)
X_test_transformed = pca.transform(X_test)

for name, label in sample_pair:
    ax.text3D(
        X_test_transformed[Y_test == label, 0].mean(),
        X_test_transformed[Y_test == label, 1].mean() + 1.5,
        X_test_transformed[Y_test == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
# y = np.choose(Y, [1, 2, 0]).astype(float)
ax.scatter(X_test_transformed[:, 0], X_test_transformed[:, 1], X_test_transformed[:, 2], c=Y_test, edgecolor="k")

# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

arr1 = np.genfromtxt('PET_Dettol_2_touching.csv', delimiter=',')
arr1 = np.delete(arr1, 10, 1)
arr1_offseted = arr1.copy()
for ar in arr1_offseted:
    ar -= ar[-1]
arr2 = np.genfromtxt('PP_tofulism_2_touching.csv', delimiter=',')
arr2 = np.delete(arr2, 10, 1)
arr2_offseted = arr2.copy()
for ar in arr2_offseted:
    ar -= ar[-1]

X = np.vstack((arr1_offseted, arr2_offseted))
sample_pair = [("PET", 1), ("PP", 5)]
Y = np.hstack((np.repeat(1, arr1_offseted.shape[0]), np.repeat(5, arr2_offseted.shape[0])))

fig = plt.figure(1, figsize=(4, 3))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in sample_pair:
    ax.text3D(
        X[Y == label, 0].mean(),
        X[Y == label, 1].mean() + 1.5,
        X[Y == label, 2].mean(),
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
# y = np.choose(Y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.nipy_spectral, edgecolor="k")

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

plt.show()

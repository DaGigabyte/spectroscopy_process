import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing
import joblib

arr1 = np.genfromtxt('PET_淳茶舍_5LEDs_vertical.csv', delimiter=',')
arr1 = np.delete(arr1, 10, 1)
arr1_offseted = arr1.copy()
for ar in arr1_offseted:
    ar -= ar[-1]
arr2 = np.genfromtxt('PP_unknown_5LEDs_vertical.csv', delimiter=',')
arr2 = np.delete(arr2, 10, 1)
arr2_offseted = arr2.copy()
for ar in arr2_offseted:
    ar -= ar[-1]

X = np.vstack((arr1_offseted, arr2_offseted))
sc = preprocessing.StandardScaler()
X = sc.fit(X)
joblib.dump(sc, "trained_sc.joblib")
pca = decomposition.PCA(n_components=3)
pca.fit(X)
joblib.dump(pca, "trained_pca.joblib")
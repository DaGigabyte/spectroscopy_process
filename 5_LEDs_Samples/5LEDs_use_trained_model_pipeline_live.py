import serial
portN = "COM7"
bps = 115200
timeOut = 5
ser = serial.Serial(portN, bps, timeout=timeOut)
print(ser.name)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing, pipeline, svm, model_selection
import joblib

def interpretSamples(data1, data2):
    X = np.vstack((data1, data2))
    Y = np.hstack((np.repeat(1, data1.shape[0]), np.repeat(5, data2.shape[0])))
    return X, Y

def pca_plot(data1, data2, figure_num = 1, title_name = "Default", standard_scaler = False):
    X, Y = interpretSamples(data1=data1, data2=data2)
    if standard_scaler:
        sc = preprocessing.StandardScaler()
        X = sc.fit_transform(X)
    sample_pair = [("PET", 1), ("PP", 5)]

    fig = plt.figure(figure_num, figsize=(4, 3))
    fig.suptitle(title_name)
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in sample_pair:
        ax.text3D(
            X[Y == label, 0].mean(),
            X[Y == label, 1].mean(),
            X[Y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(Y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, edgecolor="k")

    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])

def pca_2dplot(data1, data2, figure_num = 1, title_name = "Default", standard_scaler = False):
    X, Y = interpretSamples(data1=data1, data2=data2)
    if standard_scaler:
        sc = preprocessing.StandardScaler()
        X = sc.fit_transform(X)
    sample_pair = [("PET", 1), ("PP", 5)]

    fig = plt.figure(figure_num, figsize=(4, 3))
    fig.suptitle(title_name)
    ax = fig.add_subplot(111)
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in sample_pair:
        ax.text(
            X[Y == label, 0].mean(),
            X[Y == label, 1].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(Y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], c=Y, edgecolor="k")

pipe = joblib.load('3classes_pipe.joblib')

# fig, ax = plt.subplots()
while True:
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.array(list(map(float, num_arr)))
    arr = arr[np.array([0, 2, 3, 6, 8])]
    print(pipe.predict(arr.reshape(1,-1)))
    # print(arr)
    # ax.clear()
    # ax.set_ylim(0, 2500)
    # ax.scatter(np.arange(np.shape(arr)[0]), arr)
    # plt.pause(1e-4)

# pca_2dplot(arr1_offseted, arr2_offseted, figure_num = 100, title_name = "Original")
# pca_plot(arr1_offseted, arr2_offseted, figure_num = 1, title_name = "Original")
# pca_plot(arr1_offseted+100, arr2_offseted, figure_num = 20, title_name = "Shifted")
# pca_plot(arr1_offseted, arr2_offseted, figure_num = 2, title_name = "Normalised all", standard_scaler=True)
# pca_2dplot(arr1_offseted, arr2_offseted, figure_num = 101, title_name = "Normalised all", standard_scaler=True)

# plt.show()

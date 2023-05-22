import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing, pipeline, model_selection
import csv
import os

Empty_Re_dir = "Empty_Re"
Empty_Trans_dir = "Empty_Trans"
PET_Trans_dir = "PET_Trans"
PP_Trans_dir = "PP_Trans"
PET_Re_dir = "PET_Re"
PP_Re_dir = "PP_Re"

def dir_as_list(dir):
    output = list()
    for filename in os.listdir(dir):
        filename = dir + "\\" + filename
        with open(filename, 'r') as file:
            my_list = list(csv.reader(file, delimiter=','))
            output += [row[:-1] for row in my_list]
    return output

def dictOfList_as_dataset(my_dict):
    X, Y = [], []
    for key, value in my_dict.items():
        X.extend(value)
        Y.extend([key] * len(value))
    return np.asarray(X), np.asarray(Y)

def EmptyOrNotPipeline(X, Y):
    pipe = pipeline.Pipeline([('svc', svm.SVC(gamma='auto', kernel='linear', probability=True)), ], verbose=True)
    pipe.fit(X, Y)
    return pipe

def classifyPipeline(X, Y):
    pipe = pipeline.Pipeline([('sc', preprocessing.StandardScaler()), ('pca', decomposition.PCA(n_components=3)), ('svc', svm.SVC(gamma='auto', kernel='linear', probability=True))], verbose=True)
    pipe.fit(X, Y)
    return pipe

def listToPipe(empty_list, PET_list, PP_list):
    Re_withEmpty = {'Empty': empty_list, 'PET': PET_list, 'PP': PP_list}
    Re = {'PET': PET_list, 'PP': PP_list}
    X, Y = dictOfList_as_dataset(Re_withEmpty)
    Y_emptyornot = np.array(['Empty' if label == 'Empty' else 'Non-Empty' for label in Y])
    # print("EmptyOrNotPipeline")
    # empty_or_not_pipe = EmptyOrNotPipeline(X, Y_emptyornot)
    X, Y = dictOfList_as_dataset(Re)
    print("classifyPipeline", np.shape(Y))
    classify_pipe = classifyPipeline(X, Y)
    return None, classify_pipe

def plot_3d_pts(X, Y, labels, fig, subplot_pos):
    ax = fig.add_subplot(subplot_pos, projection="3d", elev=48, azim=134)
    ax.title.set_text('With normalisation')
    for label in labels:
        ax.text3D(
            X[Y == label, 0].mean(),
            X[Y == label, 1].mean() + 1.5,
            X[Y == label, 2].mean(),
            label,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(Y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, edgecolor="k")

    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])

    plt.show()

fig = plt.figure(1)
plot_3d_pts(XXX, fig, 121)

Empty_list = [x + y for x, y in zip(dir_as_list(Empty_Re_dir), dir_as_list(Empty_Trans_dir))]
PET_list = [x + y for x, y in zip(dir_as_list(PET_Re_dir), dir_as_list(PET_Trans_dir))]
PP_list = [x + y for x, y in zip(dir_as_list(PP_Re_dir), dir_as_list(PP_Trans_dir))]

combined_empty_or_not_pipe, combined_classify_pipe = listToPipe(Empty_list, PET_list, PP_list)
combined_classify_pipe.fit()

Re_empty_or_not_pipe, Re_classify_pipe = listToPipe(dir_as_list(Empty_Re_dir), dir_as_list(PET_Re_dir), dir_as_list(PP_Re_dir))
Trans_empty_or_not_pipe, Trans_classify_pipe = listToPipe(dir_as_list(Empty_Trans_dir), dir_as_list(PET_Trans_dir), dir_as_list(PP_Trans_dir))
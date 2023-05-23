### NEED TO BE FIXED###

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

def two_dir_as_list(dir_Re, dir_Trans):
    list1 = list()
    list2 = list()
    output = list()
    for filename1, filename2 in zip(os.listdir(dir_Re), os.listdir(dir_Trans)):
        filename1 = dir_Re + "\\" + filename1
        filename2 = dir_Trans + "\\" + filename2
        with open(filename1, 'r') as file:
            list1 = list(csv.reader(file, delimiter=','))
        with open(filename2, 'r') as file:
            list2 = list(csv.reader(file, delimiter=','))
        output += [x+y for x,y in zip([row[:-1] for row in list1], [row[:-1] for row in list1])]
    return output

def dictOfList_as_dataset(my_dict):
    X, Y = [], []
    for key, value in my_dict.items():
        if key == 'PET':
            key = 1
        elif key == 'PP':
            key = 5
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

# def dirToTransformedXY((PET_Re_dir, PET_Trans_dir), (PP_Re_dir, PP_Trans_dir)):
#     Re = {'PET': dir_as_list(PET_Re_dir), 'PP': dir_as_list(PP_Re_dir)}
#     Trans = {'PET': dir_as_list(PET_Trans_dir), 'PP': dir_as_list(PP_Trans_dir)}
#     X_r, Y_r = dictOfList_as_dataset(Re)
#     X_t, Y_t = dictOfList_as_dataset(Trans)
#     X_combined, Y_combined = np.append(X_r, X_t, 1), np.append(Y_r, Y_t, 1)
#     print("classifyPipeline", np.shape(Y_combined))
#     classify_pipe = pipeline.Pipeline([('sc', preprocessing.StandardScaler()), ('pca', decomposition.PCA(n_components=3)), ], verbose=True)
#     classify_pipe.fit(X_combined, Y_combined)


def plot_3d_pts(X, Y, labels, fig, subplot_pos):
    ax = fig.add_subplot(subplot_pos, projection="3d", elev=48, azim=134)
    ax.title.set_text('With normalisation')
    print(Y)
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


combined = {'PET': two_dir_as_list(PET_Re_dir, PET_Trans_dir), 'PP': two_dir_as_list(PP_Re_dir, PP_Trans_dir)}
# Trans = {'PET': dir_as_list(PET_Trans_dir), 'PP': dir_as_list(PP_Trans_dir)}
X_combined, Y_combined = dictOfList_as_dataset(combined)
# X_t, Y_t = dictOfList_as_dataset(Trans)
# X_combined, Y_combined = np.append(X_r, X_t, 1), np.append(Y_r, Y_t, 1)
print("classifyPipeline", np.shape(Y_combined))
classify_pipe = pipeline.Pipeline([('sc', preprocessing.StandardScaler()), ('pca', decomposition.PCA(n_components=3)), ], verbose=True)
classify_pipe.fit_transform(X_combined, Y_combined)
fig = plt.figure(1)
plot_3d_pts(X_combined, Y_combined, (1, 5), fig, 111)

# Empty_list = [x + y for x, y in zip(dir_as_list(Empty_Re_dir), dir_as_list(Empty_Trans_dir))]
# PET_list = [x + y for x, y in zip(dir_as_list(PET_Re_dir), dir_as_list(PET_Trans_dir))]
# PP_list = [x + y for x, y in zip(dir_as_list(PP_Re_dir), dir_as_list(PP_Trans_dir))]

# combined_empty_or_not_pipe, combined_classify_pipe = listToPipe(Empty_list[:len(Empty_list)//2], PET_list[:len(PET_list)//2], PP_list[:len(PP_list)//2])
# combined_classify_pipe.fit()

# Re_empty_or_not_pipe, Re_classify_pipe = listToPipe(dir_as_list(Empty_Re_dir), dir_as_list(PET_Re_dir), dir_as_list(PP_Re_dir))
# Trans_empty_or_not_pipe, Trans_classify_pipe = listToPipe(dir_as_list(Empty_Trans_dir), dir_as_list(PET_Trans_dir), dir_as_list(PP_Trans_dir))
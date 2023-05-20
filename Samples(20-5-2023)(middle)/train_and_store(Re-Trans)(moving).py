import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing, pipeline, svm, model_selection
import joblib
import csv
import os

Empty_Re_dir = "Empty_Re"
Empty_Trans_dir = "Empty_Trans"
PET_Trans_dir = "PET_Trans"
PP_Trans_dir = "PP_Trans"
PET_Re_dir = "PET_Re"
PP_Re_dir = "PP_Re"

def dir_as_dictOfList(dir):
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

def dirsToPipe(empty_dir, PET_dir, PP_dir):
    Re_withEmpty = {'Empty': dir_as_dictOfList(empty_dir), 'PET': dir_as_dictOfList(PET_dir), 'PP': dir_as_dictOfList(PP_dir)}
    Re = {'PET': dir_as_dictOfList(PET_dir), 'PP': dir_as_dictOfList(PP_dir)}
    X, Y = dictOfList_as_dataset(Re_withEmpty)
    Y_emptyornot = np.array(['Empty' if label == 'Empty' else 'Non-Empty' for label in Y])
    print("EmptyOrNotPipeline")
    empty_or_not_pipe = EmptyOrNotPipeline(X, Y_emptyornot)
    X, Y = dictOfList_as_dataset(Re)
    print("classifyPipeline", np.shape(Y))
    classify_pipe = classifyPipeline(X, Y)
    return empty_or_not_pipe, classify_pipe

Re_empty_or_not_pipe, Re_classify_pipe = dirsToPipe(Empty_Re_dir, PET_Re_dir, PP_Re_dir)
Trans_empty_or_not_pipe, Trans_classify_pipe = dirsToPipe(Empty_Trans_dir, PET_Trans_dir, PP_Trans_dir)
joblib.dump(((Re_empty_or_not_pipe, Re_classify_pipe), (Trans_empty_or_not_pipe, Trans_classify_pipe)), 'trained_pipes(Re-Trans)(Slanted)(proba)(Re-Trans-packed)(unnormalised)(3d)(middle)(2)(with_norm).joblib')
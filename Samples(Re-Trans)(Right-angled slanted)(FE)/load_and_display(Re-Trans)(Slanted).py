import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing
import joblib
import serial
portN = "COM7"
bps = 115200
timeOut = 5
ser = serial.Serial(portN, bps, timeout=timeOut)
print(ser.name)

trans_index = np.array([1, 4, 5, 7])
reflectance_index = np.array([0, 2, 3, 6, 8])
(empty_or_not_pipe, classify_pipe) = joblib.load(
    'trained_pipes(Re-Trans)(Slanted)(proba).joblib')

while True:
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.array(list(map(float, num_arr)))
    arr = arr[reflectance_index]
    print(["{:.3f}".format(p) for p in empty_or_not_pipe.predict_proba(arr.reshape(1, -1))[0]], ["{:.3f}".format(p) for p in classify_pipe.predict_proba(arr.reshape(1, -1))[0]])
    # print(empty_or_not_pipe.predict_proba(arr.reshape(1, -1)), classify_pipe.predict_proba(arr.reshape(1, -1)))
    # print(empty_or_not_pipe.predict(arr.reshape(1, -1)), classify_pipe.predict(arr.reshape(1, -1)))

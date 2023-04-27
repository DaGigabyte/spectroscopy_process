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

sc = joblib.load("trained_sc.joblib")
pca = joblib.load("trained_pca.joblib")

fig, ax = plt.subplots()
while True:
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.array(list(map(float, num_arr)))
    
    print(arr)
    ax.clear()
    ax.set_ylim(0, 2500)
    ax.scatter(np.arange(np.shape(arr)[0]), arr)
    plt.pause(1e-4)
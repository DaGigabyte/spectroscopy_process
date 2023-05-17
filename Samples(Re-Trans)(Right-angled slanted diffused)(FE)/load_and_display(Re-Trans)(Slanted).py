import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing
import joblib
import serial
ser = serial.Serial(port="COM7", baudrate=115200, timeout=5)
print(ser.name)

trans_index = np.array([1, 4, 5, 7])
reflectance_index = np.array([0, 2, 3, 6, 8])
# (empty_or_not_pipe, classify_pipe) = joblib.load('trained_pipes(Re-Trans)(Slanted)(proba).joblib')
((Re_empty_or_not_pipe, Re_classify_pipe), (Trans_empty_or_not_pipe, Trans_classify_pipe)) = joblib.load('trained_pipes(Re-Trans)(Slanted)(proba)(Re-Trans-packed)(unnormalised)(3d)(varying distance).joblib')

while True:
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.array(list(map(float, num_arr)))
    arr_Re = arr[reflectance_index]
    arr_Trans = arr[trans_index]
    print(Re_empty_or_not_pipe.classes_, Re_classify_pipe.classes_, Trans_empty_or_not_pipe.classes_, Trans_classify_pipe.classes_, end='\t')
    print("Re:", ["{:.3f}".format(p) for p in Re_empty_or_not_pipe.predict_proba(arr_Re.reshape(1, -1))[0]], ["{:.3f}".format(p) for p in Re_classify_pipe.predict_proba(arr_Re.reshape(1, -1))[0]], end=' ')
    print("Trans:", ["{:.3f}".format(p) for p in Trans_empty_or_not_pipe.predict_proba(arr_Trans.reshape(1, -1))[0]], ["{:.3f}".format(p) for p in Trans_classify_pipe.predict_proba(arr_Trans.reshape(1, -1))[0]])
    fig, axs = plt.subplots(2,2)
    axs[0,0].bar(Re_empty_or_not_pipe.classes_, Re_empty_or_not_pipe.predict_proba(arr_Re.reshape(1, -1))[0])
    plt.pause(0.001)
    
    
    # print(empty_or_not_pipe.predict_proba(arr.reshape(1, -1)), classify_pipe.predict_proba(arr.reshape(1, -1)))
    # print(empty_or_not_pipe.predict(arr.reshape(1, -1)), classify_pipe.predict(arr.reshape(1, -1)))

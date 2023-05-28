import numpy as np
from sklearn import decomposition, preprocessing
import joblib
import serial
ser = serial.Serial(port="COM7", baudrate=115200, timeout=5)
print(ser.name)

trans_index = np.array([1, 4, 5, 7])
reflectance_index = np.array([0, 2, 3, 6, 8])
# (empty_or_not_pipe, classify_pipe) = joblib.load('trained_pipes(Re-Trans)(Slanted)(proba).joblib')
(Both_empty_or_not_pipe, Both_classify_pipe) = joblib.load('trained_pipes(28-5-2023)(Both).joblib')

pipe_containers = dict()
pipe_containers['Both_empty'] = Both_empty_or_not_pipe
pipe_containers['Both_classify'] = Both_classify_pipe

str_data = ''
while not ser.in_waiting:
    pass
while True:
    while ser.in_waiting:
        str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr_both = np.array(list(map(float, num_arr[0:9])))
    arr_Re = arr_both[reflectance_index]
    arr_Trans = arr_both[trans_index]
    arrs = [arr_both, arr_both, arr_both[reflectance_index], arr_both[reflectance_index],  arr_both[trans_index], arr_both[trans_index]]
    # print(Re_empty_or_not_pipe.classes_, Re_classify_pipe.classes_, Trans_empty_or_not_pipe.classes_, Trans_classify_pipe.classes_, end='\t')
    # print("Both:", ["{:.3f}".format(p) for p in Both_empty_or_not_pipe.predict_proba(arr_both.reshape(1, -1))[0]], ["{:.3f}".format(p) for p in Both_classify_pipe.predict_proba(arr_both.reshape(1, -1))[0]], end=' ')
    # # print("Re:", ["{:.3f}".format(p) for p in Re_empty_or_not_pipe.predict_proba(arr_Re.reshape(1, -1))[0]], ["{:.3f}".format(p) for p in Re_classify_pipe.predict_proba(arr_Re.reshape(1, -1))[0]], end=' ')
    # # print("Trans:", ["{:.3f}".format(p) for p in Trans_empty_or_not_pipe.predict_proba(arr_Trans.reshape(1, -1))[0]], ["{:.3f}".format(p) for p in Trans_classify_pipe.predict_proba(arr_Trans.reshape(1, -1))[0]])
    # for container, pipe, arr in zip(bar_containers.values(), pipe_containers.values(), arrs):
    #     print(pipe)
    #     for rect, h in zip(container, pipe.predict_proba(arr.reshape(1, -1))[0]):
    #         rect.set_height(h)
    
    
    # print(empty_or_not_pipe.predict_proba(arr.reshape(1, -1)), classify_pipe.predict_proba(arr.reshape(1, -1)))
    # print(empty_or_not_pipe.predict(arr.reshape(1, -1)), classify_pipe.predict(arr.reshape(1, -1)))
    empty_or_not_prediction = Both_empty_or_not_pipe.predict(arr_both.reshape(1, -1))[0]
    classify_prediction = Both_classify_pipe.predict(arr_both.reshape(1, -1))[0]
    if (empty_or_not_prediction == 'Empty'):
        print(empty_or_not_prediction, end='\t')
    else:
        print(classify_prediction, end='\t')
    # print(Both_empty_or_not_pipe.predict(arr_both.reshape(1, -1)), Both_classify_pipe.predict(arr_both.reshape(1, -1)), end='\t')
    print(input_arr)

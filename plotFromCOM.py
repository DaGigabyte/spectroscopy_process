import serial
import numpy as np
import matplotlib.pyplot as plt
portN = "COM7"
bps = 115200
timeOut = 5
ser = serial.Serial(portN, bps, timeout=timeOut)
print(ser.name)

plt.ion()
fig, ax = plt.subplots()
while True:
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.array(list(map(float, num_arr)))
    print(arr)
    ax.scatter(np.arange(np.shape(arr)[0]), arr)
import serial
import numpy as np
import matplotlib.pyplot as plt
portN = "COM7"
bps = 115200
timeOut = 5
ser = serial.Serial(portN, bps, timeout=timeOut)
print(ser.name)

wavelengths = ("1060", "1200", "1250", "1300", "1350", "1400", "1450", "1500", "1550", "Dark(Control)")
# trans_wavelengths = ("1200", "1350", "1400", "1500")
trans_wavelengths = (wavelengths[1], wavelengths[4], wavelengths[5], wavelengths[7])
# reflectance_wavelengths = ("1060", "1250", "1300", "1450", "1550")
reflectance_wavelengths = (wavelengths[0], wavelengths[2], wavelengths[3], wavelengths[6], wavelengths[8])
trans_index = np.array([1,4,5,7])
reflectance_index = np.array([0, 2, 3, 6, 8])

fig, ax = plt.subplots(2)
while True:
    # ser.reset_input_buffer()
    while ser.in_waiting:
        print(ser.in_waiting)
        str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.array(list(map(float, num_arr)))
    arr_trans = arr[trans_index]
    arr_reflectance = arr[reflectance_index]
    print(arr)
    ax[0].clear()
    ax[0].title.set_text('Transmittance')
    ax[0].set_ylim(0, 4000)
    ax[0].scatter(trans_wavelengths, arr_trans)
    ax[1].clear()
    ax[1].title.set_text('Reflectance')
    ax[1].set_ylim(0, 4000)
    ax[1].scatter(reflectance_wavelengths, arr_reflectance)
    plt.pause(1e-4)
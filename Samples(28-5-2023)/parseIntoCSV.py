import serial
import os.path
import numpy as np
portN = "COM7"
bps = 115200
timeOut = 5
ser = serial.Serial(portN, bps, timeout=timeOut)
print(ser.name)
type_name = "PET_"
sub_dir_Tr = type_name + "Trans"
sub_dir_Re = type_name + "Re"
sub_dir_Both = type_name + "Both"
try:
    os.mkdir(os.mkdir(sub_dir_Re))
except Exception:
    pass
try:
    os.mkdir(os.mkdir(sub_dir_Tr))
except Exception:
    pass
try:
    os.mkdir(os.mkdir(sub_dir_Both))
except Exception:
    pass

dataFileName = input("Enter file name:")
dataFileName_Re = os.path.join(sub_dir_Re, dataFileName + "_Re" + ".csv")
dataFileName_Tr = os.path.join(sub_dir_Tr,dataFileName + "_Tr" + ".csv")
dataFileName_both = os.path.join(sub_dir_Both, dataFileName + ".csv")
wavelengths = ("1060", "1200", "1250", "1300", "1350", "1400", "1450", "1500", "1550", "Dark(Control)")
trans_wavelengths = (wavelengths[1], wavelengths[4], wavelengths[5], wavelengths[7])
reflectance_wavelengths = (wavelengths[0], wavelengths[2], wavelengths[3], wavelengths[6], wavelengths[8])
trans_index = np.array([1,4,5,7])
reflectance_index = np.array([0, 2, 3, 6, 8])
ser.reset_input_buffer()
for counter in range(500):
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    arr = np.asarray(num_arr)
    arr = arr[0:9]
    arr_trans = arr[trans_index]
    arr_reflectance = arr[reflectance_index]
    print(counter, num_arr)
    with open(dataFileName_Tr, 'a') as f:
        line = ""
        for n in arr_trans:
            line += n
            line += ","
        f.writelines(line)
        f.writelines('\n')
    with open(dataFileName_Re, 'a') as f:
        line = ""
        for n in arr_reflectance:
            line += n
            line += ","
        f.writelines(line)
        f.writelines('\n')
    with open(dataFileName_both, 'a') as f:
        line = ""
        for n in arr:
            line += n
            line += ","
        f.writelines(line)
        f.writelines('\n')
import serial
import os.path
portN = "COM7"
bps = 115200
timeOut = 5
ser = serial.Serial(portN, bps, timeout=timeOut)
print(ser.name)
sub_dir = "5_LEDs_Samples"
try:
    os.mkdir(sub_dir)
except Exception:
    pass
dataFileName = input("Enter file name:")
dataFileName = os.path.join(sub_dir, dataFileName + ".csv")
ser.reset_input_buffer()

while True:
    str_data = ser.readline().strip().decode()
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    print(num_arr)
    with open(dataFileName, 'a') as f:
        line = ""
        for n in num_arr:
            line += n
            line += ","
        f.writelines(line)
        f.writelines('\n')
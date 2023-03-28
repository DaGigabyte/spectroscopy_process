import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

portN = "COM7"
bps = 115200
timeOut = 5

ser = serial.Serial(portN, bps, timeout=timeOut)

print(ser.name)
plt.ion()

def animate(i, dataList):
    str_data = ser.readline().strip().decode()
    #print(str_data)
    input_arr = str_data.split("\t")
    num_arr = [n.split(':')[1] for n in input_arr]
    print(num_arr)
    plt.cla()
    for i in range(len(num_arr)):
        plt.scatter(i, num_arr[i])

fig = plt.figure()
ax = fig.add_subplot(111)

ani = animation.FuncAnimation()
# while True:
#     try:
#         str_data = ser.readline().strip().decode()
#         print(str_data)
#         input_arr = str_data.split("\t")
#         num_arr = [n.split(':')[1] for n in input_arr]
#         print(num_arr)
#         for i in range(len(num_arr)):
#             plt.scatter(i, num_arr[i])
#         plt.pause(0.05)
#     except Exception as e:
#         print("[-] Error: ", e)
import numpy as np
import matplotlib.pyplot as plt

arr1 = np.genfromtxt('input.csv', delimiter=',')
arr1 = np.delete(arr1, 10, 1)
arr1_offseted = arr1.copy()
for ar in arr1_offseted:
    ar -= ar[-1]
arr2 = np.genfromtxt('input2.csv', delimiter=',')
arr2 = np.delete(arr2, 10, 1)
arr2_offseted = arr2.copy()
for ar in arr2_offseted:
    ar -= ar[-1]

std_before = [np.std(arr1[:,i]) for i in range(10)]
std_offseted = [np.std(arr1_offseted[:,i]) for i in range(10)]

std_both = np.vstack((std_before, std_offseted))
print(std_both)

fig, (ax1, ax2) = plt.subplots(2)
ax1.scatter(x=np.tile(np.arange(arr1.shape[1]), arr1.shape[0]), y = arr1, color='r')
ax2.scatter(x=np.tile(np.arange(arr2.shape[1]), arr2.shape[0]), y = arr2, color='b')
fig2, ax3 = plt.subplots()
ax3.scatter(x=np.tile(np.arange(arr1.shape[1]), arr1.shape[0]), y = arr1, color='r')
ax3.scatter(x=np.tile(np.arange(arr2.shape[1]), arr2.shape[0]), y = arr2, color='b')
fig3, ax10 = plt.subplots()
ax10.scatter(x=np.tile(np.arange(arr1_offseted.shape[1]), arr1_offseted.shape[0]), y = arr1_offseted, color='r')
ax10.scatter(x=np.tile(np.arange(arr2_offseted.shape[1]), arr2_offseted.shape[0]), y = arr2_offseted, color='b')
plt.show()
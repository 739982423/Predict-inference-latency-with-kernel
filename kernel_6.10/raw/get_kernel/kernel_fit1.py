import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math

filename1 = "test1.csv"
filename2 = "test2.csv"
filename3 = "test3.csv"

x1 = []
y1 = []
with open(filename1, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        x1.append(float(row[1]))
        y1.append(float(row[0]))

x2 = []
y2 = []
with open(filename2, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        x2.append(float(row[1]))
        y2.append(float(row[0]))

# x3 = []
# y3 = []
# with open(filename3, mode="r", encoding="utf-8-sig") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         print(row)
#         x3.append(float(row[1]))
#         y3.append(float(row[0]))


b1 = [1] * len(x1)
b2 = [8] * len(x2)
# b3 = [16] * len(x3)

# y = y1 + y2 + y3
y = y1 + y2
x = [x1 + x2, b1 + b2]


def x2_fit(x, a, b, c):
    return a * x * x + b * x + c

def x_div2_fit(x, k, b1, b2, b3):
    return (k * x[1]) / (x[0] * x[1] * b1 + b2) + b3

def x_div2_fit2(x, k, b2, b3):
    return k / (x + b2) + b3

# x = x1 + x2 + x3
popt, pcov = curve_fit(x_div2_fit, x, y, maxfev=50000)
print(popt)


plot_point_nums = int(max(x1) - min(x1))
x_data_0 = np.linspace(100, int(max(x2)), int(max(x2)) - 100)

x_data_b1 = []
x_data_b8 = []
x_data_b16 = []
for i in range(len(x_data_0)):
    x_data_b1.append([x_data_0[i], 1])

for i in range(len(x_data_0)):
    x_data_b8.append([x_data_0[i], 8])
#
# for i in range(len(x_data_0)):
#     x_data_b16.append([x_data_0[i], 16])

y_data_b1 = [x_div2_fit(x, popt[0], popt[1], popt[2], popt[3]) for x in x_data_b1]
y_data_b8 = [x_div2_fit(x, popt[0], popt[1], popt[2], popt[3]) for x in x_data_b8]
# y_data_b16 = [x_div2_fit(x, popt[0], popt[1], popt[2], popt[3]) for x in x_data_b16]
plt.plot(x_data_0, y_data_b1, color = "green", label = "b1_curve")
plt.plot(x_data_0, y_data_b8, color = "tomato", label = "b8_curve")
# plt.plot(x_data_0, y_data_b16, color = "skyblue", label = "b16_curve")

# y_data = [x_div2_fit2(x, popt[0], popt[1], popt[2]) for x in x_data_0]
# plt.plot(x_data_0, y_data, color = "green", label = "normal_fit")

plt.plot(x1, y1, 'o', color = "green", label = "b1")
plt.plot(x2, y2, 'o', color = "tomato", label = "b8")
# plt.plot(x3, y3, 'o', color = "skyblue", label = "b16")
plt.legend()
plt.show()

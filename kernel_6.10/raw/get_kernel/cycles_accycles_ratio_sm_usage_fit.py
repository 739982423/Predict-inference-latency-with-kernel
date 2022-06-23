# 寻找总cycles与active cycles的映射关系，就目前来看，与sm使用率有关，sm率越高，active cycles越接近总cycles
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

filename = "test3.csv"

x1 = []
y1 = []
with open(filename, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        x1.append(float(row[0]))
        y1.append(float(row[1]))

def x_div2_fit2(x, k, b2, b3):
    return k / (x + b2) + b3

popt, pcov = curve_fit(x_div2_fit2, x1, y1, maxfev=50000)

x_for_plot = np.linspace(1, 100, 100)
y_for_plot = [x_div2_fit2(x, popt[0], popt[1], popt[2]) for x in x_for_plot]

plt.plot(x_for_plot, y_for_plot, color = "tomato", label = "fitted")
plt.plot(x1, y1, 'o', color = "skyblue")
plt.show()
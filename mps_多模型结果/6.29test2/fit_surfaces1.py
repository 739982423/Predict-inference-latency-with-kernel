import csv
import collections
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# 拟合某个模型m1和另一个模型m2的L2 向 dram 传输数据量与 m1受影响程度的曲面
def func(x, k11, k21, b1, b2, b3):
    return ((k11 * x[0] + b1) / (k21 * x[1]  + b2)) * x[0] + b3

# 保存m1(受影响模型)的传输数据量的文件
m1_data_csv1 = "C:\\Users\\73998\Desktop\实验数据\kernel_6.10\\raw\get_kernel\hit_rate_message.csv"
# 保存m2(影响m1的模型)的传输数据量的文件
m2_data_csv1 = "./tmp_r_v.csv"

# 首先读取m1的传输数据到哈希表
m1_data = collections.defaultdict(float)
with open(m1_data_csv1, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if len(row) > 0 and row[0] != "model name":
            bs = row[1]
            gpu = row[2]
            message = float(row[12]) / 1000 / 10000
            m1_data[row[0] + "_b" + row[2] + "_g" + row[1]] = message

x_data1 = [[] for i in range(2)]
y_data1 = []

with open(m2_data_csv1, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) <= 38 or row[38] == "":
            continue
        # 获取m1传输数据信息
        model_name = row[0]
        gpu = row[1]
        bs = row[2]
        m1_key = row[0] + "_b" + row[1] + "_g" + row[2]
        m1_transfer_data = m1_data[m1_key]
        # print(row)
        # 获取m2传输数据信息和y轴的值
        m2_transfer_data = float(row[38]) / 1000 / 10000
        influenced_percent = float(row[6]) * 100

        # 添加到拟合的x和y中
        if m1_transfer_data == 0:
            continue
        x_data1[0].append(m1_transfer_data)
        x_data1[1].append(m2_transfer_data)
        y_data1.append(influenced_percent)

plot_x = []
for i in range(len(x_data1[0])):
    print("--------------")
    print(x_data1[0][i])
    print(x_data1[1][i])
    print(y_data1[i])
    plot_x.append(x_data1[1][i] / x_data1[0][i])

plt.plot(plot_x, y_data1, 'o')
plt.show()
# print(len(x_data1), len(y_data1))

# popt, pcov = curve_fit(func, x_data1, y_data1, maxfev=3000000)
# print(popt)

# resnet g10 b8 data=108200221
# vgg19  g50 b16 data=703953538.2

# x = [0, 0]
# x[0] = 103484228/1000/10000
# tmp_m2 = [54520003.21,
# 37183213.61,
# 45816348.2,
# 40877042.23,
# 143832417.5,
# 103966610.2,
# 130623931.9,
# 128885991.4,
# 231434250,
# 206815793,
# 281088647.8,
# 287020323,
# 295361424,
# 287577230,
# 410549446,
# 431675828.3]
#
#
# print(popt)
# for i in range(len(tmp_m2)):
#     x[1] = tmp_m2[i]/1000/10000
#     influenced = func(x, popt[0], popt[1], popt[2], popt[3], popt[4])
#     print(influenced)


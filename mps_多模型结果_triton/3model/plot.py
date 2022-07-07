import matplotlib.pyplot as plt
import csv
import numpy as np

csv_name1 = "res_th_vgg19_resnet50_densenet201.csv"
model_nums = 3

# 初始化
line_idx = 0
data_list = []

# 定义画图部分的数据
res = []

with open(csv_name1, mode = "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    # 每部分数据占用的行数 = 标签一行 + 标题一行 + 每个模型占用一行 + 空一行
    data_lines = 2 + model_nums + 1

    # 开始读取数据
    for row in reader:

        data_list.append(row)
        line_idx += 1

        # 取出了一个数据段
        if line_idx == data_lines:
            # data_list[2] - data_list[2 + (x - 1)]是要读取的x个模型的数据
            # 其中:
            # data_list[2][0] = 模型名称
            # data_list[2][1] = GPU分配量
            # data_list[2][2] = batch
            # data_list[2][3] = server端latency
            # data_list[2][4] = clinet端latency
            # data_list[2][5] = server端吞吐量
            # data_list[2][6] = client端吞吐量
            res.append((data_list[2], data_list[3], data_list[4]))
            # 重新初始化
            line_idx = 0
            data_list = []

for row in res:
    print(row)

r_g10_b1_d_g10_b1_latency = []
r_g10_b1_d_g10_b8_latency = []
r_g10_b8_d_g10_b1_latency = []
r_g10_b8_d_g10_b8_latency = []

r_g40_b1_d_g10_b8_latency = []
r_g40_b8_d_g10_b8_latency = []
r_g10_b8_d_g40_b1_latency = []
r_g10_b8_d_g40_b8_latency = []


for m1, m2, m3 in res:
    # m1:vgg19, m2:resnet50, m3:densenet201
    if m1[1] == '40' and m2[1] == '10' and m2[2] == '1' and m3[1] == '10' and m3[2] == '1':
        r_g10_b1_d_g10_b1_latency.append((m1[3],m1[2]))
    if m1[1] == '40' and m2[1] == '10' and m2[2] == '1' and m3[1] == '10' and m3[2] == '8':
        r_g10_b1_d_g10_b8_latency.append((m1[3],m1[2]))
    if m1[1] == '40' and m2[1] == '10' and m2[2] == '8' and m3[1] == '10' and m3[2] == '1':
        r_g10_b8_d_g10_b1_latency.append((m1[3],m1[2]))
    if m1[1] == '40' and m2[1] == '10' and m2[2] == '8' and m3[1] == '10' and m3[2] == '8':
        r_g10_b8_d_g10_b8_latency.append((m1[3],m1[2]))

    if m1[1] == '40' and m2[1] == '40' and m2[2] == '1' and m3[1] == '10' and m3[2] == '8':
        r_g40_b1_d_g10_b8_latency.append((m1[3],m1[2]))
    if m1[1] == '40' and m2[1] == '40' and m2[2] == '8' and m3[1] == '10' and m3[2] == '8':
        r_g40_b8_d_g10_b8_latency.append((m1[3],m1[2]))

    if m1[1] == '40' and m2[1] == '10' and m2[2] == '8' and m3[1] == '40' and m3[2] == '1':
        r_g10_b8_d_g40_b1_latency.append((m1[3],m1[2]))
    if m1[1] == '40' and m2[1] == '10' and m2[2] == '8' and m3[1] == '40' and m3[2] == '8':
        r_g10_b8_d_g40_b8_latency.append((m1[3],m1[2]))


print(r_g10_b1_d_g10_b1_latency)
print(r_g10_b1_d_g10_b8_latency)
print(r_g10_b8_d_g10_b1_latency)
print(r_g10_b8_d_g10_b8_latency)
print(r_g40_b8_d_g10_b8_latency)
print(r_g40_b1_d_g10_b8_latency)
print(r_g10_b8_d_g40_b8_latency)
print(r_g10_b8_d_g40_b1_latency)

plt.grid(alpha = 0.8, linestyle = '-.')
y_name = ["r_g10_b1_d_g10_b1_latency", "r_g10_b1_d_g10_b8_latency", "r_g10_b8_d_g10_b1_latency", "r_g10_b8_d_g10_b8_latency",
          "r_g40_b8_d_g10_b8_latency", "r_g40_b1_d_g10_b8_latency", "r_g10_b8_d_g40_b8_latency", "r_g10_b8_d_g40_b1_latency"]
color = ['#ff7e44','#EE82EE','#3ccbff', '#7ee64c','#a980ff','tomato','#40E0D0','#fdd814']
for i in range(8):
    tmp_x = [1, 8, 16]
    tmp_y = [int(eval(y_name[i])[j][0]) / 1000 for j in range(3)]
    plt.plot(tmp_x, tmp_y, 'o--', color=color[i], label=y_name[i][:-8])
vgg19_gpu40_single_latency = [7.735, 25.330, 43.668, 49.743, 53.173]
vgg19_gpu40_single_bs = [1, 4, 8, 12, 16]
plt.plot(vgg19_gpu40_single_bs, vgg19_gpu40_single_latency, 'o--', color = "#000000", label = "Origin")
plt.xlabel("VGG19 batchsize (40% GPU resource).")
plt.ylabel("VGG19 Real Latency(ms).")
plt.legend()
plt.show()

# 查表
origin_v_g40_b8 = 234.347

v_g10_b8_r_g10_b8 = 233.771
v_g10_b8_r_g40_b8 = 253.061

v_g10_b8_r_g10_b8_d_g10_b8 = 234.990
v_g10_b8_r_g10_b8_d_g40_b8 = 250.205

v_g10_b8_r_g40_b8_d_g10_b8 = 256.592
v_g10_b8_r_g40_b8_d_g40_b8 = 274.021

plt.subplot(2, 1, 1)
plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot([i for i in range(0, 9)], [origin_v_g40_b8] * 9, '-.', color = 'tomato', label = 'Origin')
plt.plot(1, origin_v_g40_b8, 'o', color = 'tomato', markersize=8)
plt.plot(2, v_g10_b8_r_g10_b8, 'x', color = '#3ccbff', markersize=8, label = 'r_g10_b8')
plt.plot(3, v_g10_b8_r_g40_b8 , 'x', color = '#3ccbff', markersize=8, label = 'r_g40_b8')
plt.plot(4, v_g10_b8_r_g10_b8_d_g10_b8, '*', color = '#a980ff', markersize=10, label = "r_g10_b8_d_g10_b8")
plt.plot(5, v_g10_b8_r_g10_b8_d_g40_b8, '*', color = '#a980ff', markersize=10, label = "r_g10_b8_d_g40_b8")
plt.plot(6, v_g10_b8_r_g40_b8_d_g10_b8, '*', color = '#a980ff', markersize=10, label = "r_g40_b8_d_g10_b8")
plt.plot(7, v_g10_b8_r_g40_b8_d_g40_b8, '*', color = '#a980ff', markersize=10, label = "r_g40_b8_d_g40_b8")

plt.xlabel("co-Location Type.")
plt.ylabel("VGG19 Real Latency(ms).")
plt.axis([0.5, 7.5, 220, 280])

num1 = 1.03
num2 = 0.25
num3 = 3
num4 = 0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

plt.subplot(2, 1, 2)
plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot([i for i in range(0, 9)], [1] * 9, '-.', color = 'tomato', label = 'Origin')
plt.bar(1, 1.0, color = 'tomato' ,width=0.4)
plt.bar(2, v_g10_b8_r_g10_b8 / origin_v_g40_b8, color = '#3ccbff' ,width=0.4, label = 'r_g10_b8')
plt.bar(3, v_g10_b8_r_g40_b8 / origin_v_g40_b8, color = '#3ccbff' ,width=0.4, label = 'r_g40_b8')
plt.bar(4, v_g10_b8_r_g10_b8_d_g10_b8 / origin_v_g40_b8, color = '#a980ff' ,width=0.4, label = 'r_g10_b8_d_g10_b8')
plt.bar(5, v_g10_b8_r_g10_b8_d_g40_b8 / origin_v_g40_b8 , color = '#a980ff' ,width=0.4, label = 'r_g10_b8_d_g40_b8')
plt.bar(6, v_g10_b8_r_g40_b8_d_g10_b8 / origin_v_g40_b8, color = '#a980ff' ,width=0.4, label = 'r_g40_b8_d_g10_b8')
plt.bar(7, v_g10_b8_r_g40_b8_d_g40_b8 / origin_v_g40_b8, color = '#a980ff' ,width=0.4, label = 'r_g40_b8_d_g40_b8')
plt.xlabel("co-Location Type.")
plt.ylabel("co-Location Latency / Origin Latency.")
plt.axis([0.5, 7.5, 0.85, 1.2])
num1 = 1.03
num2 = 0.25
num3 = 3
num4 = 0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
plt.show()

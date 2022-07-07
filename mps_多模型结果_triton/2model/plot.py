import matplotlib.pyplot as plt
import csv
import numpy as np

csv_name1 = "res_th_vgg19_resnet50.csv"
csv_name2 = "res_th_densenet201_resnet50.csv"
model_nums = 2

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
            res.append((data_list[2], data_list[3]))
            # 重新初始化
            line_idx = 0
            data_list = []

# ----------图1，vgg19(固定40%GPU资源占用)的latency在不同resnet占用GPU资源和bs的情况下，与vgg19的bs关系的折线图-----------

# 第一个模型在GPU分配40%，bs=1时的latency
m1_gpu40_b1_latency = []
# 对应的第二个模型的GPU和BS情况
m1_gpu40_b1_flag = []

# 第一个模型在GPU分配40%，bs=1时的latency
m1_gpu40_b8_latency = []
# 对应的第二个模型的GPU和BS情况
m1_gpu40_b8_flag = []

# 第一个模型在GPU分配40%，bs=1时的latency
m1_gpu40_b16_latency = []
# 对应的第二个模型的GPU和BS情况
m1_gpu40_b16_flag = []

for m1, m2 in res:
    if m1[1] == '40' and m1[2] == '1':
        m1_gpu40_b1_latency.append(int(m1[3]) / 1000)
        m1_gpu40_b1_flag.append(m2[0] + "_GPU_" + m2[1] + "%_b" + m2[2])
    if m1[1] == '40' and m1[2] == '8':
        m1_gpu40_b8_latency.append(int(m1[3]) / 1000)
        m1_gpu40_b8_flag.append(m2[0] + "_GPU_" + m2[1] + "%_b" + m2[2])
    if m1[1] == '40' and m1[2] == '16':
        m1_gpu40_b16_latency.append(int(m1[3]) / 1000)
        m1_gpu40_b16_flag.append(m2[0] + "_GPU_" + m2[1] + "%_b" + m2[2])

print(m1_gpu40_b1_latency)
print(m1_gpu40_b1_flag)
print(m1_gpu40_b8_latency)
print(m1_gpu40_b8_flag)
print(m1_gpu40_b16_latency)
print(m1_gpu40_b16_flag)

vgg19_gpu40_single_latency = [7.735, 25.330, 43.668, 49.743, 53.173]
vgg19_gpu40_single_bs = [1, 4, 8, 12, 16]
plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot(vgg19_gpu40_single_bs, vgg19_gpu40_single_latency, 'o--', color = "#000000", label = "Origin")
color = ['tomato', '#ff7e44', '#fdd814', '#7ee64c', '#3ccbff', '#a980ff']
for i in range(len(m1_gpu40_b1_flag)):
    tmp_x = [1, 8, 16]
    tmp_y = [m1_gpu40_b1_latency[i], m1_gpu40_b8_latency[i], m1_gpu40_b16_latency[i]]
    plt.plot(tmp_x, tmp_y, 'o--', color = color[i], label = m1_gpu40_b1_flag[i])

plt.xlabel("VGG19 batchsize (40% GPU resource).")
plt.ylabel("VGG19 Real Latency(ms).")
plt.legend()
plt.show()

# ----------- 图2，vgg19(固定bs=8)的latency在不同resnet占用GPU资源和bs的情况下，与GPU资源关系的折线图 ---------------
m1_gpu10_b8_latency = []
m1_gpu10_b8_flag = []
m1_gpu40_b8_latency = []
m1_gpu40_b8_flag = []
m1_gpu80_b8_latency = []
m1_gpu80_b8_flag = []

for m1, m2 in res:
    if m1[2] == '8' and m1[1] == '10':
        m1_gpu10_b8_latency.append(int(m1[3]) / 1000)
        m1_gpu10_b8_flag.append(m2[0] + "_GPU_" + m2[1] + "%_b" + m2[2])
    if m1[2] == '8' and m1[1] == '40':
        m1_gpu40_b8_latency.append(int(m1[3]) / 1000)
        m1_gpu40_b8_flag.append(m2[0] + "_GPU_" + m2[1] + "%_b" + m2[2])
    if m1[2] == '8' and m1[1] == '80':
        m1_gpu80_b8_latency.append(int(m1[3]) / 1000)
        m1_gpu80_b8_flag.append(m2[0] + "_GPU_" + m2[1] + "%_b" + m2[2])
print("**********")
print(m1_gpu10_b8_latency, m1_gpu10_b8_flag)

vgg19_gpu10_single_latency = [28.569, 102.423, 234.347, 265.410, 271.701]
vgg19_gpu10_single_bs = [1, 4, 8, 12, 16]
x = [[1,2,3], [4,5,6], [7,8,9]]

color = ['tomato', '#ff7e44', '#fdd814', '#7ee64c', '#3ccbff', '#a980ff']
plt.subplot(2, 1, 1)
plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot(x[0], m1_gpu10_b8_latency[:3], 'o', color = color[3], markersize=8, label = "resnet50,GPU10%,b=1,8,16")
plt.plot(x[1], m1_gpu10_b8_latency[3:6], 'x', color = color[4], markersize=8, label = "resnet50,GPU40%,b=1,8,16")
plt.plot(x[2], m1_gpu10_b8_latency[6:], '*', color = color[5], markersize=8, label = "resnet50,GPU80%,b=1,8,16")
plt.plot(x[0] + x[1] + x[2], [234.347] * 9 , 'o-.', color = color[1], label = "Origin")
plt.xlabel("co-Location Type.")
plt.ylabel("VGG19 Real Latency(ms).")
plt.legend()

plt.subplot(2, 1, 2)
plt.grid(alpha = 0.8, linestyle = '-.')
for i in range(len(m1_gpu10_b8_latency)):
    if i < 3:
        color_ = color[3]
    elif i < 6:
        color_ = color[4]
    else:
        color_ = color[5]
    plt.bar(i + 1, m1_gpu10_b8_latency[i] / 234.347, color = color_,width=0.4)
plt.plot([i for i in range(11)], [1] * 11 , 'o-.', color = color[1], label = "Origin")
plt.xlabel("co-Location Type.")
plt.ylabel("co-Location Latency / Origin Latency.")
plt.xticks([1,2,3,4,5,6,7,8,9])
plt.axis([0, 10, 0.8, 1.3])
plt.show()

# print(m1_gpu40_b8_latency, m1_gpu40_b8_flag)
# print(m1_gpu80_b8_latency, m1_gpu80_b8_flag)
# vgg19_single_latency = [7.735, 25.330, 43.668, 49.743, 53.173]
# vgg19_single_bs = [1, 4, 8, 12, 16]
# plt.grid(alpha = 0.8, linestyle = '-.')
# plt.plot(vgg19_single_bs, vgg19_single_latency, 'o--', color = "#000000", label = "Origin")
# color = ['tomato', '#ff7e44', '#fdd814', '#7ee64c', '#3ccbff', '#a980ff']
# for i in range(len(m1_gpu10_b8_flag)):
#     tmp_x = [10, 40, 80]
#     tmp_y = [m1_gpu10_b8_latency[i], m1_gpu40_b8_latency[i], m1_gpu80_b8_latency[i]]
#     plt.plot(tmp_x, tmp_y, 'o--', color = color[i], label = m1_gpu10_b8_flag[i])
# plt.xlabel("VGG19 batchsize (40% GPU resource).")
# plt.ylabel("VGG19 Real Latency(ms).")
# plt.legend()
# plt.show()

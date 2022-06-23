import matplotlib.pyplot as plt
import csv
import numpy as np

csv_name1 = "res_th_vgg19.csv"      # 画图1，2
csv_name2 = "res_th_vgg19_3.csv"    # 画图3
model_nums = 1

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
            print(data_list[2])
            res.append(data_list[2])
            # 重新初始化
            line_idx = 0
            data_list = []

# 画图部分
# figure1，GPU RESOURCE固定，不同batch的latency
gpu10_bs = []
gpu10_latency = []
gpu20_latency = []
gpu40_latency = []
gpu60_latency = []
gpu80_latency = []
gpu100_latency = []
for row in res:
    if row[1] == '10':
        gpu10_bs.append(int(row[2]))
        gpu10_latency.append(int(row[3]) / 1000)
    if row[1] == '20':
        gpu20_latency.append(int(row[3]) / 1000)
    if row[1] == '40':
        gpu40_latency.append(int(row[3]) / 1000)
    if row[1] == '60':
        gpu60_latency.append(int(row[3]) / 1000)
    if row[1] == '80':
        gpu80_latency.append(int(row[3]) / 1000)
    if row[1] == '100':
        gpu100_latency.append(int(row[3]) / 1000)

plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot(gpu10_bs, gpu10_latency, 'o--', color='tomato', label = "GPU 10%")
plt.plot(gpu10_bs, gpu20_latency, 'o--', color='#ff7e44', label = "GPU 20%")
plt.plot(gpu10_bs, gpu40_latency, 'o--', color='#fdd814', label = "GPU 40%")
plt.plot(gpu10_bs, gpu60_latency, 'o--', color='#7ee64c', label = "GPU 60%")
plt.plot(gpu10_bs, gpu80_latency, 'o--', color='#3ccbff', label = "GPU 80%")
plt.plot(gpu10_bs, gpu100_latency, 'o--', color='#a980ff', label = "GPU 100%")
plt.xlabel("VGG19 batchsize.")
plt.ylabel("VGG19 Real Latency(ms).")
plt.legend()
plt.show()

# figure2 固定batch, 不同resource下的latency
gpu_resource = []
bs1_latency = []
bs4_latency = []
bs8_latency = []
bs12_latency = []
bs16_latency = []
for row in res:
    if row[2] == '1':
        gpu_resource.append(int(row[1]))
        bs1_latency.append(int(row[3]) / 1000)
    if row[2] == '4':
        bs4_latency.append(int(row[3]) / 1000)
    if row[2] == '8':
        bs8_latency.append(int(row[3]) / 1000)
    if row[2] == '12':
        bs12_latency.append(int(row[3]) / 1000)
    if row[2] == '16':
        bs16_latency.append(int(row[3]) / 1000)

plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot(gpu_resource, bs1_latency, color='tomato', label = "batch = 1")
plt.plot(gpu_resource, bs4_latency, color='#ff7e44', label = "batch = 4")
plt.plot(gpu_resource, bs8_latency, color='#fdd814', label = "batch = 8")
plt.plot(gpu_resource, bs12_latency, color='#7ee64c', label = "batch = 12")
plt.plot(gpu_resource, bs16_latency, color='#3ccbff', label = "batch = 16")
plt.xlabel("GPU resource allocated(%).")
plt.ylabel("VGG19 Real Latency(ms).")
plt.legend()
plt.show()

print("------------------------------------------------------------------------------")
# figure 3，更详细的latency与GPU资源分配量的关系走势
# 初始化
line_idx = 0
data_list = []

# 定义画图部分的数据
res = []
with open(csv_name2, mode = "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    # 每部分数据占用的行数 = 标签一行 + 标题一行 + 每个模型占用一行 + 空一行
    data_lines = 2 + model_nums + 1
    # 开始读取数据
    for row in reader:
        data_list.append(row)
        line_idx += 1
        # 取出了一个数据段
        if line_idx == data_lines:
            print(data_list[2])
            res.append(data_list[2])
            # 重新初始化
            line_idx = 0
            data_list = []

bs4_gpu_resource = []
bs8_gpu_resource = []
bs16_gpu_resource = []
bs32_gpu_resource = []
bs4_latency = []
bs8_latency = []
bs16_latency = []
bs32_latency = []
for row in res:
    if row[2] == '4':
        bs4_gpu_resource.append(int(row[1]))
        bs4_latency.append(int(row[3]) / 1000)
    if row[2] == '8':
        bs8_gpu_resource.append(int(row[1]))
        bs8_latency.append(int(row[3]) / 1000)
    if row[2] == '16':
        bs16_gpu_resource.append(int(row[1]))
        bs16_latency.append(int(row[3]) / 1000)
    if row[2] == '32':
        bs32_gpu_resource.append(int(row[1]))
        bs32_latency.append(int(row[3]) / 1000)

plt.grid(alpha = 0.8, linestyle = '-.')
plt.plot(bs4_gpu_resource, bs4_latency, 'o--', color='tomato', label = "batch = 4")
plt.plot(bs8_gpu_resource, bs8_latency, 'o--',color='#ff7e44', label = "batch = 8")
plt.plot(bs16_gpu_resource, bs16_latency, 'o--', color='#fdd814', label = "batch = 16")
plt.plot(bs32_gpu_resource, bs32_latency, 'o--', color='#7ee64c', label = "batch = 32")
plt.xlabel("GPU resource allocated(%).")
plt.ylabel("VGG19 Real Latency(ms).")
plt.legend()
plt.show()






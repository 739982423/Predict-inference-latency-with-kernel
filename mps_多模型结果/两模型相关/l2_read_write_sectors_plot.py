import csv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import collections
from mpl_toolkits import mplot3d

def func(x, k1, k2, k3, b):
    return (k1 * x[0] * x[0] + k2 * x[0]) / (k3 * x[1] + b)

def func_show(x1, x2, k1, k2, k3, b):
    z = (k1 * x1 * x1 + k2 * x1) / (k3 * x2 + b)
    return z

gpu_resource = ["10","25","50","75"]
bs = ["1", "8", "16", "32"]

# 保存m1(受影响模型)的传输数据量的文件
m1_data_csv1 = "C:\\Users\\73998\Desktop\实验数据\kernel_6.10\\raw\get_kernel\hit_rate_message.csv"

# 首先读取m1的传输数据到哈希表
m1_data = collections.defaultdict(float)
with open(m1_data_csv1, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if len(row) > 0 and row[0] != "model name":
            message = float(row[12]) / 1000 / 10000
            m1_data[row[0] + "_b" + row[1] + "_g" + row[2]] = message

res = [[0] * len(gpu_resource) for i in range(len(bs))]

# 三维散点图所需的xyz
res_3d_x_g10 = []
res_3d_y_g10 = []
res_3d_z_g10 = []
res_3d_x_g25 = []
res_3d_y_g25 = []
res_3d_z_g25 = []
res_3d_x_g50 = []
res_3d_y_g50 = []
res_3d_z_g50 = []
res_3d_x_g75 = []
res_3d_y_g75 = []
res_3d_z_g75 = []

# 拟合某GPU分配量时 x=影响模型数据传输量 y=受影响模型数据传输量 z=受影响百分比 三者的曲面 所需的xyz参数
surface_g10_x = []
surface_g10_y = []
surface_g10_z = []

surface_g25_x = []
surface_g25_y = []
surface_g25_z = []

surface_g50_x = []
surface_g50_y = []
surface_g50_z = []

surface_g75_x = []
surface_g75_y = []
surface_g75_z = []
for idx1, gpu in enumerate(gpu_resource):
    for idx2, b in enumerate(bs):
        print("g{}_b{}".format(gpu, b))
        data_csv3 = "./tmp_r_d.csv"
        x_data1 = []
        y_data1 = []

        data_csv2 = "./tmp_r_m.csv"
        # data_csv2 = "./tmp_d_m.csv"
        x_data2 = []
        y_data2 = []

        data_csv1 = "./tmp_r_v.csv"
        # data_csv1 = "./tmp_d_v.csv"
        x_data3 = []
        y_data3 = []

        compared_bs = b
        compared_gpu = gpu

        with open(data_csv1, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                # model_name = "densenet201"
                model_name = "resnet50"
                if len(row) <= 37 or row[38] == '':
                    continue
                if row[0] == model_name and row[1] == compared_gpu and row[2] == compared_bs:
                    # print(row)
                    influenced_percent = float(row[6]) * 100
                    # x = float(row[35]) / 1000 / 10000
                    x = (float(row[35]) + float(row[38])) / 1000 / 10000
                    x_data1.append(x)
                    y_data1.append(influenced_percent)

                    cur_x_3d = (float(row[35]) + float(row[38])) / 1000 / 10000    # m2 l2 dram数据交换量
                    cur_y_3d = m1_data[model_name + "_b" + compared_bs + "_g" + compared_gpu]
                    if gpu == "10":
                        res_3d_x_g10.append(cur_x_3d)
                        res_3d_y_g10.append(cur_y_3d)
                        res_3d_z_g10.append(influenced_percent)

                        surface_g10_x.append(cur_x_3d)
                        surface_g10_y.append(cur_y_3d)
                        surface_g10_z.append(influenced_percent)
                    elif gpu == "25":
                        res_3d_x_g25.append(cur_x_3d)
                        res_3d_y_g25.append(cur_y_3d)
                        res_3d_z_g25.append(influenced_percent)

                        surface_g25_x.append(cur_x_3d)
                        surface_g25_y.append(cur_y_3d)
                        surface_g25_z.append(influenced_percent)
                    elif gpu == "50":
                        res_3d_x_g50.append(cur_x_3d)
                        res_3d_y_g50.append(cur_y_3d)
                        res_3d_z_g50.append(influenced_percent)

                        surface_g50_x.append(cur_x_3d)
                        surface_g50_y.append(cur_y_3d)
                        surface_g50_z.append(influenced_percent)
                    elif gpu == "75":
                        res_3d_x_g75.append(cur_x_3d)
                        res_3d_y_g75.append(cur_y_3d)
                        res_3d_z_g75.append(influenced_percent)

                        surface_g75_x.append(cur_x_3d)
                        surface_g75_y.append(cur_y_3d)
                        surface_g75_z.append(influenced_percent)
        with open(data_csv2, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                # model_name = "densenet201"
                model_name = "resnet50"
                if len(row) <= 37 or row[38] == '':
                    continue
                if row[0] == model_name and row[1] == compared_gpu and row[2] == compared_bs:
                    influenced_percent = float(row[6]) * 100
                    # x = float(row[35]) / 1000 / 10000
                    x = (float(row[35]) + float(row[38])) / 1000 / 10000
                    x_data1.append(x)
                    y_data1.append(influenced_percent)

                    cur_x_3d = (float(row[35]) + float(row[38])) / 1000 / 10000  # m2 l2 dram数据交换量
                    cur_y_3d = m1_data[model_name + "_b" + compared_bs + "_g" + compared_gpu]
                    if gpu == "10":
                        res_3d_x_g10.append(cur_x_3d)
                        res_3d_y_g10.append(cur_y_3d)
                        res_3d_z_g10.append(influenced_percent)

                        surface_g10_x.append(cur_x_3d)
                        surface_g10_y.append(cur_y_3d)
                        surface_g10_z.append(influenced_percent)
                    elif gpu == "25":
                        res_3d_x_g25.append(cur_x_3d)
                        res_3d_y_g25.append(cur_y_3d)
                        res_3d_z_g25.append(influenced_percent)

                        surface_g25_x.append(cur_x_3d)
                        surface_g25_y.append(cur_y_3d)
                        surface_g25_z.append(influenced_percent)
                    elif gpu == "50":
                        res_3d_x_g50.append(cur_x_3d)
                        res_3d_y_g50.append(cur_y_3d)
                        res_3d_z_g50.append(influenced_percent)

                        surface_g50_x.append(cur_x_3d)
                        surface_g50_y.append(cur_y_3d)
                        surface_g50_z.append(influenced_percent)
                    elif gpu == "75":
                        res_3d_x_g75.append(cur_x_3d)
                        res_3d_y_g75.append(cur_y_3d)
                        res_3d_z_g75.append(influenced_percent)

                        surface_g75_x.append(cur_x_3d)
                        surface_g75_y.append(cur_y_3d)
                        surface_g75_z.append(influenced_percent)


        with open(data_csv3, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                # model_name = "densenet201"
                model_name = "resnet50"
                if len(row) <= 37 or row[38] == '':
                    continue
                if row[0] == model_name and row[1] == compared_gpu and row[2] == compared_bs:
                    influenced_percent = float(row[6]) * 100
                    # x = float(row[35]) / 1000 / 10000
                    x = (float(row[35]) + float(row[38])) / 1000 / 10000
                    x_data1.append(x)
                    y_data1.append(influenced_percent)

                    cur_x_3d = (float(row[35]) + float(row[38])) / 1000 / 10000  # m2 l2 dram数据交换量
                    cur_y_3d = m1_data[model_name + "_b" + compared_bs + "_g" + compared_gpu]
                    # print(model_name + "_b" + compared_bs + "_g" + compared_gpu)
                    if gpu == "10":
                        res_3d_x_g10.append(cur_x_3d)
                        res_3d_y_g10.append(cur_y_3d)
                        res_3d_z_g10.append(influenced_percent)

                        surface_g10_x.append(cur_x_3d)
                        surface_g10_y.append(cur_y_3d)
                        surface_g10_z.append(influenced_percent)
                    elif gpu == "25":
                        res_3d_x_g25.append(cur_x_3d)
                        res_3d_y_g25.append(cur_y_3d)
                        res_3d_z_g25.append(influenced_percent)

                        surface_g25_x.append(cur_x_3d)
                        surface_g25_y.append(cur_y_3d)
                        surface_g25_z.append(influenced_percent)
                    elif gpu == "50":
                        res_3d_x_g50.append(cur_x_3d)
                        res_3d_y_g50.append(cur_y_3d)
                        res_3d_z_g50.append(influenced_percent)

                        surface_g50_x.append(cur_x_3d)
                        surface_g50_y.append(cur_y_3d)
                        surface_g50_z.append(influenced_percent)
                    elif gpu == "75":
                        res_3d_x_g75.append(cur_x_3d)
                        res_3d_y_g75.append(cur_y_3d)
                        res_3d_z_g75.append(influenced_percent)

                        surface_g75_x.append(cur_x_3d)
                        surface_g75_y.append(cur_y_3d)
                        surface_g75_z.append(influenced_percent)

        # popt, pcov = curve_fit(func, x_data1 + x_data2 + x_data3, y_data1 + y_data2 + y_data3)
        # print(popt[0])
        # res[idx1][idx2] = (popt[0], popt[1])
        # x_plot = np.linspace(0, max(x_data1 + x_data2 + x_data3), 100)
        # y_plot = [func(x_plot[i], popt[0], popt[1]) for i in range(len(x_plot))]
        #
        # plt.plot(x_plot, y_plot)
        # plt.plot(x_data1, y_data1, 'o', color = 'orange')
        # plt.plot(x_data2, y_data2, 'o', color = 'tomato')
        # plt.plot(x_data3, y_data3, 'o', color = 'blue')
        # plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(np.array(res_3d_x_g10), np.array(res_3d_y_g10), np.array(res_3d_z_g10), color = "green", label = "g10")
ax.scatter3D(np.array(res_3d_x_g25), np.array(res_3d_y_g25), np.array(res_3d_z_g25), color = "orange", label = "g25")
ax.scatter3D(np.array(res_3d_x_g50), np.array(res_3d_y_g50), np.array(res_3d_z_g50), color = "red", label = "g50")
ax.scatter3D(np.array(res_3d_x_g75), np.array(res_3d_y_g75), np.array(res_3d_z_g75), color = "blue", label = "g75")
# ax.set_title('3d Scatter plot')
plt.gca().invert_xaxis()
plt.xlabel("m1 Data amount")
plt.ylabel("m2 Data amount")
plt.legend()
plt.show()
# plt.plot([10, 25, 50, 75], [res[0][0], res[1][0], res[2][0], res[3][0]], 'o',label = "b1")
# plt.plot([10, 25, 50, 75], [res[0][1], res[1][1], res[2][1], res[3][1]], 'o',label = "b8")
# plt.plot([10, 25, 50, 75], [res[0][2], res[1][2], res[2][2], res[3][2]], 'o',label = "b16")
# plt.plot([10, 25, 50, 75], [res[0][3], res[1][3], res[2][3], res[3][3]], 'o',label = "b32")
# plt.legend()
# plt.show()

# 拟合g=10的时候 x=影响模型数据传输量 y=受影响模型数据传输量 z=受影响百分比 三者的曲面
g10_surface_plot_x = [[], []]
if len(surface_g10_x) != 0:
    for i in range(len(surface_g10_x)):
        g10_surface_plot_x[0].append(surface_g10_x[i])
        g10_surface_plot_x[1].append(surface_g10_y[i])
    g10_popt, pcov = curve_fit(func, g10_surface_plot_x, surface_g10_z)
    print("gpu=10%, popt = {}".format(g10_popt))

    # 看看曲面什么样
    plot_x = np.linspace(10, 400, 400)
    plot_y = np.linspace(10, 40, 40)
    X, Y = np.meshgrid(plot_x, plot_y)
    Z = func_show(X, Y, g10_popt[0], g10_popt[1], g10_popt[2], g10_popt[3])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.gca().invert_xaxis()
    ax.plot_surface(X, Y, Z)
    ax.set_title('gpu=10% Surface plot')
    plt.show()

g25_surface_plot_x = [[], []]
if len(surface_g25_x) != 0:
    for i in range(len(surface_g25_x)):
        g25_surface_plot_x[0].append(surface_g25_x[i])
        g25_surface_plot_x[1].append(surface_g25_y[i])
    g25_popt, pcov = curve_fit(func, g25_surface_plot_x, surface_g25_z)
    print("gpu=25%, popt = {}".format(g25_popt))
    plot_x = np.linspace(10, 400, 400)
    plot_y = np.linspace(10, 40, 40)
    X, Y = np.meshgrid(plot_x, plot_y)
    Z = func_show(X, Y, g25_popt[0], g25_popt[1], g25_popt[2], g25_popt[3])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.gca().invert_xaxis()
    ax.plot_surface(X, Y, Z)
    ax.set_title('gpu=25% Surface plot')
    plt.show()

g50_surface_plot_x = [[], []]
if len(surface_g50_x) != 0:
    for i in range(len(surface_g50_x)):
        g50_surface_plot_x[0].append(surface_g50_x[i])
        g50_surface_plot_x[1].append(surface_g50_y[i])
    g50_popt, pcov = curve_fit(func, g50_surface_plot_x, surface_g50_z)
    print("gpu=50%, popt = {}".format(g50_popt))
    plot_x = np.linspace(10, 100, 100)
    plot_y = np.linspace(10, 40, 40)
    X, Y = np.meshgrid(plot_x, plot_y)
    Z = func_show(X, Y, g50_popt[0], g50_popt[1], g50_popt[2], g50_popt[3])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.gca().invert_xaxis()
    ax.plot_surface(X, Y, Z)
    ax.set_title('预测模型受影响百分比(%)')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("共存模型L2缓存相关数据交换量")
    plt.ylabel("预测模型L2缓存相关数据交换量")
    plt.show()

g75_surface_plot_x = [[], []]
if len(surface_g75_x) != 0:
    for i in range(len(surface_g75_x)):
        g75_surface_plot_x[0].append(surface_g75_x[i])
        g75_surface_plot_x[1].append(surface_g75_y[i])
    g75_popt, pcov = curve_fit(func, g75_surface_plot_x, surface_g75_z)
    print("gpu=75%, popt = {}".format(g75_popt))
    plot_x = np.linspace(10, 400, 400)
    plot_y = np.linspace(10, 40, 40)
    X, Y = np.meshgrid(plot_x, plot_y)
    Z = func_show(X, Y, g75_popt[0], g75_popt[1], g75_popt[2], g75_popt[3])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.gca().invert_xaxis()
    ax.plot_surface(X, Y, Z)
    ax.set_title('gpu=75% Surface plot')
    plt.show()


# 实际计算验证一下看看
# gpu = 10
if len(surface_g10_z) != 0:
    MAE = 0
    for i in range(len(surface_g10_z)):
        predict = func_show(g10_surface_plot_x[0][i], g10_surface_plot_x[1][i], g10_popt[0], g10_popt[1], g10_popt[2], g10_popt[3])
        real = surface_g10_z[i]
        # print("predict: {}, real: {}".format(predict, real))
        MAE += abs(real - predict)
    MAE /= len(surface_g10_z)
    print("GPU=10%, MAE = {}".format(MAE))

# gpu = 25
if len(surface_g25_z) != 0:
    MAE = 0
    for i in range(len(surface_g25_z)):
        predict = func_show(g25_surface_plot_x[0][i], g25_surface_plot_x[1][i], g25_popt[0], g25_popt[1], g25_popt[2], g25_popt[3])
        real = surface_g25_z[i]
        # print("predict: {}, real: {}".format(predict, real))
        MAE += abs(real - predict)
    MAE /= len(surface_g25_z)
    print("GPU=25%, MAE = {}".format(MAE))

# gpu = 50
if len(surface_g50_z) != 0:
    MAE = 0
    for i in range(len(surface_g50_z)):
        predict = func_show(g50_surface_plot_x[0][i], g50_surface_plot_x[1][i], g50_popt[0], g50_popt[1], g50_popt[2], g50_popt[3])
        real = surface_g50_z[i]
        # print("predict: {}, real: {}".format(predict, real))
        MAE += abs(real - predict)
    MAE /= len(surface_g50_z)
    print("GPU=50%, MAE = {}".format(MAE))

# gpu = 75
if len(surface_g75_z) != 0:
    MAE = 0
    for i in range(len(surface_g75_z)):
        predict = func_show(g75_surface_plot_x[0][i], g75_surface_plot_x[1][i], g75_popt[0], g75_popt[1], g75_popt[2], g75_popt[3])
        real = surface_g75_z[i]
        # print("predict: {}, real: {}".format(predict, real))
        MAE += abs(real - predict)
    MAE /= len(surface_g75_z)
    print("GPU=75%, MAE = {}".format(MAE))
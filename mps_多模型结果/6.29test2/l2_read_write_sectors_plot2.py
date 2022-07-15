import csv
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import collections

def func(x, k1, k2, k3, b):
    return (k1 * x[0] * x[0] + k2 * x[0]) / (k3 * x[1] + b)

def func_show(x1, x2, k1, k2, k3, b):
    z = (k1 * x1 * x1 + k2 * x1) / (k3 * x2 + b)
    return z

# 四维曲面拟合
def func_4dimension(x, k1, k2, k3, k4, b):
    return k4 * x[2] * (k1 * x[0] * x[0] + k2 * x[0]) / (k3 * x[1] + b)

gpu_resource = ["10", "25", "50", "75"]
bs = ["1", "8", "16", "32"]

# 保存m1(受影响模型)的传输数据量的文件
m_data_csv1 = "C:\\Users\\73998\Desktop\实验数据\kernel_6.10\\raw\get_kernel\hit_rate_message.csv"

# 首先读取m1的传输数据到哈希表
m_data = collections.defaultdict(float)
with open(m_data_csv1, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if len(row) > 0 and row[0] != "model name":
            message1 = float(row[9]) / 1000 / 10000     #L1与L2之间交互的的数据量
            message2 = float(row[12]) / 1000 / 10000    #L2与Dram之间交互的数据量
            m_data[row[0] + "_b" + row[1] + "_g" + row[2]] = (message1, message2)

res = [[0] * len(gpu_resource) for i in range(len(bs))]

# 四维拟合所需的各维度自变量 和 因变量
four_dimension_fit_x1 = []
four_dimension_fit_x2 = []
four_dimension_fit_x3 = []
four_dimension_fit_y = []
for idx1, gpu in enumerate(gpu_resource):
    for idx2, b in enumerate(bs):
        print("g{}_b{}".format(gpu, b))
        data_csv3 = "./tmp_r_d.csv"
        # data_csv3 = "./tmp_r_v.csv"
        # data_csv1 = "./tmp_r_m.csv"
        x_data1 = []
        y_data1 = []

        # data_csv2 = "./tmp_r_m.csv"
        data_csv2 = "./tmp_d_m.csv"
        # data_csv2 = "./tmp_v_m.csv"
        x_data2 = []
        y_data2 = []

        # data_csv1 = "./tmp_r_v.csv"
        data_csv1 = "./tmp_d_v.csv"
        # data_csv3 = "./tmp_v_m.csv"
        x_data3 = []
        y_data3 = []

        compared_bs = b
        compared_gpu = gpu

        with open(data_csv1, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                model_name = "densenet201"
                # model_name = "resnet50"
                # model_name = "vgg19"
                # model_name = "mobilenet"
                if len(row) <= 37 or row[38] == '':
                    continue
                if row[0] == model_name and row[1] == compared_gpu and row[2] == compared_bs:
                    # 因变量
                    influenced_percent = float(row[6]) * 100
                    # m2(共存的影响模型) l2 dram数据交换量
                    cur_x1_4d = (float(row[35]) + float(row[38])) / 1000 / 10000
                    # m1(受影响模型) l2 dram数据交换量
                    cur_x2_4d = m_data[model_name + "_b" + compared_bs + "_g" + compared_gpu][1] + m_data[model_name + "_b" + compared_bs + "_g" + compared_gpu][0]

                    four_dimension_fit_x1.append(cur_x1_4d)
                    four_dimension_fit_x2.append(cur_x2_4d)
                    four_dimension_fit_x3.append(float(compared_gpu))
                    four_dimension_fit_y.append(influenced_percent)
        with open(data_csv2, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                model_name = "densenet201"
                # model_name = "resnet50"
                # model_name = "vgg19"
                # model_name = "mobilenet"
                if len(row) <= 37 or row[38] == '':
                    continue
                if row[0] == model_name and row[1] == compared_gpu and row[2] == compared_bs:
                    influenced_percent = float(row[6]) * 100
                    # m2(共存的影响模型) l2 dram数据交换量
                    cur_x1_4d = (float(row[35]) + float(row[38])) / 1000 / 10000
                    # m1(受影响模型) l2 dram数据交换量
                    cur_x2_4d = m_data[model_name + "_b" + compared_bs + "_g" + compared_gpu][1] + m_data[model_name + "_b" + compared_bs + "_g" + compared_gpu][0]

                    four_dimension_fit_x1.append(cur_x1_4d)
                    four_dimension_fit_x2.append(cur_x2_4d)
                    four_dimension_fit_x3.append(float(compared_gpu))
                    four_dimension_fit_y.append(influenced_percent)


        with open(data_csv3, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                model_name = "densenet201"
                # model_name = "resnet50"
                # model_name = "vgg19"
                # model_name = "mobilenet"
                if len(row) <= 37 or row[38] == '':
                    continue
                if row[0] == model_name and row[1] == compared_gpu and row[2] == compared_bs:
                    influenced_percent = float(row[6]) * 100
                    # m2(共存的影响模型) l2 dram数据交换量
                    cur_x1_4d = (float(row[35]) + float(row[38])) / 1000 / 10000
                    # m1(受影响模型) l2 dram数据交换量
                    cur_x2_4d = m_data[model_name + "_b" + compared_bs + "_g" + compared_gpu][1] + m_data[model_name + "_b" + compared_bs + "_g" + compared_gpu][0]

                    four_dimension_fit_x1.append(cur_x1_4d)
                    four_dimension_fit_x2.append(cur_x2_4d)
                    four_dimension_fit_x3.append(float(compared_gpu))
                    four_dimension_fit_y.append(influenced_percent)

# 拟合
fit_x = [[], [], []]
for i in range(len(four_dimension_fit_y)):
    fit_x[0].append(four_dimension_fit_x1[i])
    fit_x[1].append(four_dimension_fit_x2[i])
    fit_x[2].append(four_dimension_fit_x3[i])

popt, pcov = curve_fit(func_4dimension, fit_x, four_dimension_fit_y)

fit_model_4dimension_surface = "fit_model_4dimension_surface.csv"
with open(fit_model_4dimension_surface, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["densenet201", "k1", "k2", "k3", "k4", "b"])
    writer.writerow(["", popt[0], popt[1], popt[2], popt[3], popt[4]])
#
# 验证
MAE = 0
for i in range(len(four_dimension_fit_y)):
    x1 = fit_x[0][i]
    x2 = fit_x[1][i]
    x3 = fit_x[2][i]
    predict = func_4dimension([x1, x2, x3], popt[0], popt[1], popt[2], popt[3], popt[4])
    print(x1, x2, x3)
    real = four_dimension_fit_y[i]
    MAE += abs(predict - real)
    print("predict = {}, real = {}".format(predict, real))

print("MAE = {}".format(MAE / len(four_dimension_fit_y)))

MAE = 0
n = 0
input_csv = "./tmp_d_v.csv"
with open(input_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 14 and row[0] == "densenet201" and row[1] != "origin" and row[9] != "":
            m1_name = row[0]
            m1_g = int(row[1])
            m1_b = int(row[2])
            m1_looptimes = int(row[4])
            m1_real_latency_increased = float(row[6])

            m2_name = row[8]
            m2_g = int(row[9])
            m2_b = int(row[10])
            m2_looptimes = int(row[12])
            m2_real_latency_increased = float(row[14])

            if m2_g == 90:
                continue
            base_m1_data_transfer = m_data[m1_name + "_b" + str(m1_b) + "_g" + str(m1_g)]
            base_m2_data_transfer = m_data[m2_name + "_b" + str(m2_b) + "_g" + str(m2_g)]
            if base_m1_data_transfer == 0 or base_m2_data_transfer == 0:
                continue
            ratio = m2_looptimes / m1_looptimes
            x0 = (base_m2_data_transfer[0] + base_m2_data_transfer[1]) * ratio
            x1 = base_m1_data_transfer[1] + base_m1_data_transfer[0]
            x2 = m1_g

            predict = func_4dimension([x0, x1, x2], popt[0], popt[1], popt[2], popt[3], popt[4])
            # print(row)
            # print(x0, x1, x2)
            print("predict = {}, real = {}".format(predict, m1_real_latency_increased * 100))
            MAE += abs(predict - m1_real_latency_increased * 100)
            n += 1
print("MAE = {}".format(MAE / n))
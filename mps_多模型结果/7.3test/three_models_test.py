import csv
import collections

# 四维曲面拟合
def func_4dimension(x, k1, k2, k3, k4, b):
    # x[0]:共存模型的数据交换量 x[1]:受影响模型的数据交换量 x[2]:受影响模型的GPU分配量
    return k4 * x[2] * (k1 * x[0] * x[0] + k2 * x[0]) / (k3 * x[1] + b)


m1_name = "densenet201"
m2_name = "resnet50"
m3_name = "vgg19"

resnet50_coefficient = []
densenet201_coefficient = []
vgg19_coefficient = []

with open("./coefficient.csv", mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == "resnet50":
            resnet50_coefficient.append(float(row[1]))
            resnet50_coefficient.append(float(row[2]))
            resnet50_coefficient.append(float(row[3]))
            resnet50_coefficient.append(float(row[4]))
            resnet50_coefficient.append(float(row[5]))
        if row[0] == "densenet201":
            densenet201_coefficient.append(float(row[1]))
            densenet201_coefficient.append(float(row[2]))
            densenet201_coefficient.append(float(row[3]))
            densenet201_coefficient.append(float(row[4]))
            densenet201_coefficient.append(float(row[5]))
        if row[0] == "vgg19":
            vgg19_coefficient.append(float(row[1]))
            vgg19_coefficient.append(float(row[2]))
            vgg19_coefficient.append(float(row[3]))
            vgg19_coefficient.append(float(row[4]))
            vgg19_coefficient.append(float(row[5]))
print(resnet50_coefficient,
densenet201_coefficient,
vgg19_coefficient)
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
            m_data[row[0] + "_b" + row[1] + "_g" + row[2]] = message1 + message2

# 打开真实共存结果的文件，读取其中不同的模型组合
input_csv = "./tmp_" + m1_name[0] + "_" + m2_name[0] + "_" + m3_name[0] + ".csv"
MAE1 = 0
MAE2 = 0
MAE3 = 0
ignore = 0
n = 0
with open(input_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        # if n >= 1:
        #     break
        # print(row)

        m1_name = row[0]
        m1_g = int(row[1])
        m1_b = int(row[2])
        m1_looptimes = int(row[4])
        m1_latency_increase = float(row[6]) * 100
        # print("m1 loop times:", m1_looptimes)

        m2_name = row[8]
        m2_g = int(row[9])
        m2_b = int(row[10])
        m2_looptimes = int(row[12])
        m2_latency_increase = float(row[14]) * 100
        # print("m2 loop times:", m2_looptimes)

        m3_name = row[16]
        m3_g = int(row[17])
        m3_b = int(row[18])
        m3_looptimes = int(row[20])
        m3_latency_increase = float(row[22]) * 100
        # print("m3 loop times:", m3_looptimes)

        if m1_b == 4 or m2_b == 4 or m3_b == 4:
            continue
        # ------------------------------ 预测数据准备 -----------------------------------
        print("--------")
        print(row)
        # m1,m2,m3的base数据传输量（未归一化时）
        base_m1_data_transfer = m_data[m1_name + "_b" + str(m1_b) + "_g" + str(m1_g)]
        base_m2_data_transfer = m_data[m2_name + "_b" + str(m2_b) + "_g" + str(m2_g)]
        base_m3_data_transfer = m_data[m3_name + "_b" + str(m3_b) + "_g" + str(m3_g)]
        # print("base m2 数据传输量:{}".format(base_m2_data_transfer))
        # print("base m3 数据传输量:{}".format(base_m3_data_transfer))

        # predict m1
        # m2归一化后的数据传输量
        m2_ratio = m2_looptimes / m1_looptimes
        normalized_m2_data_transfer = base_m2_data_transfer * m2_ratio
        # print("normalized m2 数据传输量:{}".format(normalized_m2_data_transfer))

        # m3归一化后的数据传输量
        m3_ratio = m3_looptimes / m1_looptimes
        normalized_m3_data_transfer = base_m3_data_transfer * m3_ratio
        # print("normalized m3 数据传输量:{}".format(normalized_m3_data_transfer))

        x0 = normalized_m2_data_transfer + normalized_m3_data_transfer
        x1 = base_m1_data_transfer
        x2 = m1_g
        coff = eval(m1_name + "_coefficient")
        predict = func_4dimension([x0, x1, x2], coff[0], coff[1], coff[2], coff[3], coff[4])
        MAE1 += abs(predict - m1_latency_increase)
        print("predict m1:{}, real m1:{}, g1:{}".format(predict, m1_latency_increase, m1_g))



        # predict m2
        # m1归一化后的数据传输量
        m1_ratio = m1_looptimes / m2_looptimes
        normalized_m1_data_transfer = base_m1_data_transfer * m1_ratio
        # print("normalized m1 数据传输量:{}".format(normalized_m1_data_transfer))

        # m3归一化后的数据传输量
        m3_ratio = m3_looptimes / m2_looptimes
        normalized_m3_data_transfer = base_m3_data_transfer * m3_ratio
        # print("normalized m3 数据传输量:{}".format(normalized_m3_data_transfer))

        x0 = normalized_m1_data_transfer + normalized_m3_data_transfer
        x1 = base_m2_data_transfer
        x2 = m2_g
        coff = eval(m2_name + "_coefficient")
        predict = func_4dimension([x0, x1, x2], coff[0], coff[1], coff[2], coff[3], coff[4])
        MAE2 += abs(predict - m2_latency_increase)
        print("predict m2:{}, real m2:{}, g2:{}".format(predict, m2_latency_increase, m2_g))


        # predict m3
        # m1归一化后的数据传输量
        m1_ratio = m1_looptimes / m3_looptimes
        normalized_m1_data_transfer = base_m1_data_transfer * m1_ratio
        # print("normalized m1 数据传输量:{}".format(normalized_m1_data_transfer))

        # m2归一化后的数据传输量
        m2_ratio = m2_looptimes / m3_looptimes
        normalized_m2_data_transfer = base_m2_data_transfer * m2_ratio
        # print("normalized m2 数据传输量:{}".format(normalized_m2_data_transfer))

        x0 = normalized_m1_data_transfer + normalized_m2_data_transfer
        x1 = base_m3_data_transfer
        x2 = m3_g
        coff = eval(m3_name + "_coefficient")
        predict = func_4dimension([x0, x1, x2], coff[0], coff[1], coff[2], coff[3], coff[4])
        # if abs(predict - m3_latency_increase) >= 10:
        #     ignore += 1
        #     n += 1
        #     continue
        MAE3 += abs(predict - m3_latency_increase)
        print("predict m3:{}, real m3:{}, g3:{}".format(predict, m3_latency_increase, m3_g))
        n += 1
print("MAE1 = {}".format(MAE1 / n))
print("MAE2 = {}".format(MAE2 / n))
print("MAE3 = {}".format(MAE3 / n))
print(ignore, n)
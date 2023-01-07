import csv
import collections

m1_name = "densenet201"
m2_name = "vgg19"
m3_name = "resnet50"
m4_name = "mobilenet"


real_test_res_csv = "./../../tmp_d_m_r_v.csv"
predict_res_csv = "./densenet201_vgg19_resnet50_mobilenet.csv"

m1_origin_csv = "./../../../base/4_models_base/" + m1_name[0] + ".csv"
m2_origin_csv = "./../../../base/4_models_base/" + m2_name[0] + ".csv"
m3_origin_csv = "./../../../base/4_models_base/" + m3_name[0] + ".csv"
m4_origin_csv = "./../../../base/4_models_base/" + m4_name[0] + ".csv"

# 读取四个模型单独运行时的latency, 以便之后计算误差百分比
m1_origin_latency = []
m2_origin_latency = []
m3_origin_latency = []
m4_origin_latency = []

# 读取第一个模型的单独运行结果
with open(m1_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m1_origin_latency.append(row)

# 读取第二个模型的单独运行结果
with open(m2_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m2_origin_latency.append(row)

# 读取第三个模型的单独运行结果
with open(m3_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m3_origin_latency.append(row)

# 读取第四个模型的单独运行结果
with open(m4_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m4_origin_latency.append(row)

# 读取预测的结果
predict_res = []
with open(predict_res_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        predict_res.append(row)
bs = [1, 8, 16]
gpu = [10, 25, 50, 75]
# 整理预测的结果
predict_res_hash = collections.defaultdict(list)
for i in range(0, len(predict_res), 6):
    if predict_res[i][1] == m1_name:
        m1_gpu = int(predict_res[i + 1][1])
        m2_gpu = int(predict_res[i + 1][2])
        m3_gpu = int(predict_res[i + 1][3])
        m4_gpu = int(predict_res[i + 1][4])
        m1_bs = int(predict_res[i + 2][1])
        m2_bs = int(predict_res[i + 2][2])
        m3_bs = int(predict_res[i + 2][3])
        m4_bs = int(predict_res[i + 2][4])
        m1_latency = float(predict_res[i + 4][1])
        m2_latency = float(predict_res[i + 4][2])
        m3_latency = float(predict_res[i + 4][3])
        m4_latency = float(predict_res[i + 4][4])
        # print(predict_res[i],predict_res[i+1],predict_res[i+2],predict_res[i+3])
        m1_single_latency = float(m1_origin_latency[gpu.index(m1_gpu)][bs.index(m1_bs)])
        m2_single_latency = float(m2_origin_latency[gpu.index(m2_gpu)][bs.index(m2_bs)])
        m3_single_latency = float(m3_origin_latency[gpu.index(m3_gpu)][bs.index(m3_bs)])
        m4_single_latency = float(m4_origin_latency[gpu.index(m4_gpu)][bs.index(m4_bs)])

        key = m1_name + "_g" + str(m1_gpu) + "_b" + str(m1_bs) + "_" + m2_name + "_g" + str(m2_gpu) + "_b" + str(m2_bs) + "_" \
              + m3_name + "_g" + str(m3_gpu) + "_b" + str(m3_bs) + "_" + m4_name + "_g" + str(m4_gpu) + "_b" + str(m4_bs)

        m1_error = abs((m1_latency - m1_single_latency) / m1_single_latency) * 100
        m2_error = abs((m2_latency - m2_single_latency) / m2_single_latency) * 100
        m3_error = abs((m3_latency - m3_single_latency) / m3_single_latency) * 100
        m4_error = abs((m4_latency - m4_single_latency) / m4_single_latency) * 100

        value = ((m1_latency, m2_latency, m3_latency, m4_latency), (m1_error, m2_error, m3_error, m4_error), (m1_single_latency, m2_single_latency, m3_single_latency, m4_single_latency))
        predict_res_hash[key] = value

print(predict_res_hash)
# 读取真实测试的结果
MAE1 = 0
MAE2 = 0
MAE3 = 0
MAE4 = 0
n = 0
ratio = 0.6
output = open("./4models_predict_output.txt", "w")
# 用一个list保存所有预测结果的误差
errors = []
with open(real_test_res_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m1_name = row[0]
        m1_gpu = int(row[1])
        m1_bs = int(row[2])
        m1_real_latency = float(row[3])
        m1_latency_increase = float(row[6]) * 100

        m4_name = row[8]
        m4_gpu = int(row[9])
        m4_bs = int(row[10])
        m4_real_latency = float(row[11])
        m4_latency_increase = float(row[14]) * 100

        m3_name = row[16]
        m3_gpu = int(row[17])
        m3_bs = int(row[18])
        m3_real_latency = float(row[19])
        m3_latency_increase = float(row[22]) * 100

        m2_name = row[24]
        m2_gpu = int(row[25])
        m2_bs = int(row[26])
        m2_real_latency = float(row[27])
        m2_latency_increase = float(row[30]) * 100
        # 获取预测的值
        predict_key = m1_name + "_g" + str(m1_gpu) + "_b" + str(m1_bs) + "_" + m2_name + "_g" + str(m2_gpu) + "_b" + \
                      str(m2_bs) + "_" + m3_name + "_g" + str(m3_gpu) + "_b" + str(m3_bs) + "_" + m4_name + "_g" + \
                      str(m4_gpu) + "_b" + str(m4_bs)

        value = predict_res_hash[predict_key]

        if value == []:
            continue
        # 得到当前测试配置下的预测结果
        predict_m1_latency = value[0][0]
        predict_m2_latency = value[0][1]
        predict_m3_latency = value[0][2]
        predict_m4_latency = value[0][3]

        predict_m1_latency_increase = value[1][0]
        predict_m2_latency_increase = value[1][1]
        predict_m3_latency_increase = value[1][2]
        predict_m4_latency_increase = value[1][3]

        single_latency1 = value[2][0]
        single_latency2 = value[2][1]
        single_latency3 = value[2][2]
        single_latency4 = value[2][3]

        print("------------------------------------------------------------------------------", file=output)
        # 计算MAE
        print(predict_key, file=output)
        print("single latency m1:{}".format(single_latency1), file=output)
        print("共存predict m1:{}, 共存real m1:{}".format(predict_m1_latency, m1_real_latency), file=output)
        print("预测百分比增长 m1:{}, 真实百分比增长 m1:{}".format(predict_m1_latency_increase, m1_latency_increase), file=output)
        MAE1 += abs(predict_m1_latency_increase - m1_latency_increase) * ratio
        errors.append(abs(predict_m1_latency_increase - m1_latency_increase))

        print("***", file=output)
        print("single latency m2:{}".format(single_latency2), file=output)
        print("共存predict m2:{}, 共存real m2:{}".format(predict_m2_latency, m2_real_latency), file=output)
        print("预测百分比增长 m2:{}, 真实百分比增长 m2:{}".format(predict_m2_latency_increase, m2_latency_increase), file=output)
        MAE2 += abs(predict_m2_latency_increase - m2_latency_increase) * ratio
        errors.append(abs(predict_m2_latency_increase - m2_latency_increase))

        print("***", file=output)
        print("single latency m3:{}".format(single_latency3), file=output)
        print("共存predict m3:{}, 共存real m3:{}".format(predict_m3_latency, m3_real_latency), file=output)
        print("预测百分比增长 m3:{}, 真实百分比增长 m3:{}".format(predict_m3_latency_increase, m3_latency_increase), file=output)
        MAE3 += abs(predict_m3_latency_increase - m3_latency_increase) * ratio
        errors.append(abs(predict_m3_latency_increase - m3_latency_increase))

        print("***", file=output)
        print("single latency m4:{}".format(single_latency4), file=output)
        print("共存predict m4:{}, 共存real m4:{}".format(predict_m4_latency, m4_real_latency), file=output)
        print("预测百分比增长 m4:{}, 真实百分比增长 m4:{}".format(predict_m4_latency_increase, m4_latency_increase), file=output)
        MAE4 += abs(predict_m4_latency_increase - m4_latency_increase) * ratio
        errors.append(abs(predict_m4_latency_increase - m4_latency_increase))

        n += 1

print("MAE1 = {}".format(MAE1 / n))
print("MAE2 = {}".format(MAE2 / n))
print("MAE3 = {}".format(MAE3 / n))
print("MAE4 = {}".format(MAE4 / n))

errors_result_file = "./errors8.csv"
with open(errors_result_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    for error in errors:
        writer.writerow([error])
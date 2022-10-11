# 拟合 GPU经历的周期数与SM活跃的周期数的比值 与 SM利用率 的乘积
import csv
import collections
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


models = ["resnet50", "vgg19", "densenet201", "mobilenet"]
batch = [1, 8, 16, 32]

gpu10_ratio_mul_smusage_hash = collections.defaultdict(list)
gpu25_ratio_mul_smusage_hash = collections.defaultdict(list)
gpu50_ratio_mul_smusage_hash = collections.defaultdict(list)
gpu75_ratio_mul_smusage_hash = collections.defaultdict(list)
gpu100_ratio_mul_smusage_hash = collections.defaultdict(list)

def get_number(s):
    s_list = s.split(",")
    res = 0
    cur_power = 0
    for i in range(len(s_list) - 1, -1, -1):
        res += float(s_list[i]) * pow(1000, cur_power)
        cur_power += 1
    return res


for m in models:
    for b in batch:
        file_name = "filtered_raw_" + m + "_b" + str(b) + "_g10.csv"
        with open(file_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "ID":
                    kernel_name = row[2]
                    kernel_instructions = row[10]
                    kernel_type = (kernel_name, kernel_instructions)
                    total_cycles = get_number(row[6])
                    active_cycles = get_number(row[8])
                    ratio = total_cycles / active_cycles
                    sm_usage_rate = float(row[14])
                    gpu10_ratio_mul_smusage_hash[kernel_type].append(sm_usage_rate * ratio)

for m in models:
    for b in batch:
        file_name = "filtered_raw_" + m + "_b" + str(b) + "_g25.csv"
        with open(file_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "ID":
                    kernel_name = row[2]
                    kernel_instructions = row[10]
                    kernel_type = (kernel_name, kernel_instructions)
                    total_cycles = get_number(row[6])
                    active_cycles = get_number(row[8])
                    ratio = total_cycles / active_cycles
                    sm_usage_rate = float(row[14])
                    gpu25_ratio_mul_smusage_hash[kernel_type].append(sm_usage_rate * ratio)

for m in models:
    for b in batch:
        file_name = "filtered_raw_" + m + "_b" + str(b) + "_g50.csv"
        with open(file_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "ID":
                    kernel_name = row[2]
                    kernel_instructions = row[10]
                    kernel_type = (kernel_name, kernel_instructions)
                    total_cycles = get_number(row[6])
                    active_cycles = get_number(row[8])
                    ratio = total_cycles / active_cycles
                    sm_usage_rate = float(row[14])
                    gpu50_ratio_mul_smusage_hash[kernel_type].append(sm_usage_rate * ratio)

for m in models:
    for b in batch:
        file_name = "filtered_raw_" + m + "_b" + str(b) + "_g75.csv"
        with open(file_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "ID":
                    kernel_name = row[2]
                    kernel_instructions = row[10]
                    kernel_type = (kernel_name, kernel_instructions)
                    total_cycles = get_number(row[6])
                    active_cycles = get_number(row[8])
                    ratio = total_cycles / active_cycles
                    sm_usage_rate = float(row[14])
                    gpu75_ratio_mul_smusage_hash[kernel_type].append(sm_usage_rate * ratio)

for m in models:
    for b in batch:
        file_name = "filtered_raw_" + m + "_b" + str(b) + "_g100.csv"
        with open(file_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != "ID":
                    kernel_name = row[2]
                    kernel_instructions = row[10]
                    kernel_type = (kernel_name, kernel_instructions)

                    total_cycles = get_number(row[6])
                    active_cycles = get_number(row[8])
                    ratio = total_cycles / active_cycles
                    sm_usage_rate = float(row[14])
                    gpu100_ratio_mul_smusage_hash[kernel_type].append(sm_usage_rate * ratio)

for k, v in gpu10_ratio_mul_smusage_hash.items():
    C = 0
    for num in v:
        C += float(num)
    average_C = C / len(v)
    gpu10_ratio_mul_smusage_hash[k] = average_C

for k, v in gpu25_ratio_mul_smusage_hash.items():
    C = 0
    for num in v:
        C += float(num)
    average_C = C / len(v)
    gpu25_ratio_mul_smusage_hash[k] = average_C

for k, v in gpu50_ratio_mul_smusage_hash.items():
    C = 0
    for num in v:
        C += float(num)
    average_C = C / len(v)
    gpu50_ratio_mul_smusage_hash[k] = average_C

for k, v in gpu75_ratio_mul_smusage_hash.items():
    C = 0
    for num in v:
        C += float(num)
    average_C = C / len(v)
    gpu75_ratio_mul_smusage_hash[k] = average_C

for k, v in gpu100_ratio_mul_smusage_hash.items():
    C = 0
    for num in v:
        C += float(num)
    average_C = C / len(v)
    gpu100_ratio_mul_smusage_hash[k] = average_C


# 统计所有kernel的类型
total_kernel_types = collections.defaultdict(lambda:[-1,-1,-1,-1,-1])
for k, v in gpu10_ratio_mul_smusage_hash.items():
    origin = total_kernel_types[k]
    origin[0] = v
    total_kernel_types[k] = origin

for k, v in gpu25_ratio_mul_smusage_hash.items():
    origin = total_kernel_types[k]
    origin[1] = v
    total_kernel_types[k] = origin

for k, v in gpu50_ratio_mul_smusage_hash.items():
    origin = total_kernel_types[k]
    origin[2] = v
    total_kernel_types[k] = origin

for k, v in gpu75_ratio_mul_smusage_hash.items():
    origin = total_kernel_types[k]
    origin[3] = v
    total_kernel_types[k] = origin

for k, v in gpu100_ratio_mul_smusage_hash.items():
    origin = total_kernel_types[k]
    origin[4] = v
    total_kernel_types[k] = origin

#
# def func1(x, k, b1, b2):
#     r = k / (x + b1) + b2
#     return r


def func2(x, k, b):
    r = k * x + b
    return r

kernel_kx_hash = collections.defaultdict(list)

for k, v in total_kernel_types.items():
    print('------------')
    print(k)
    print(v)
    candidate_x_data = [10, 25, 50, 75, 100]
    x_data = []
    y_data = []
    for idx, val in enumerate(v):
        if val != -1:
            x_data.append(candidate_x_data[idx])
            y_data.append(val)
    if len(x_data) >= 3:
        popt, pcov = curve_fit(func2, x_data, y_data, maxfev=500000)
        kernel_kx_hash[k] = (1, popt)
    else:
        kernel_kx_hash[k] = (0, sum(y_data) / len(y_data))

# res_file = "../predict/kernel_actcycles_cycles_map.csv"
# with open(res_file, mode="w", encoding="utf-8-sig", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["kernel_name", "instructions", "map_k", "map_b1", "map_b2", "no_kx_onlyC"])
#
#     for k, v in sorted(kernel_kx_hash.items()):
#         kernel_name = k[0]
#         instructions = k[1]
#         row = []
#         if v[0] == 0:
#             row = [kernel_name, instructions, "", "", "", v[1]]
#         elif v[0] == 1:
#             row = [kernel_name, instructions, v[1][0], v[1][1], v[1][2]]
#         writer.writerow(row)

res_file = "../predict/kernel_actcycles_cycles_map2.csv"
with open(res_file, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel_name", "instructions", "map_k", "map_b1", "no_kx_onlyC"])

    for k, v in sorted(kernel_kx_hash.items()):
        kernel_name = k[0]
        instructions = k[1]
        row = []
        if v[0] == 0:
            row = [kernel_name, instructions, "", "", v[1]]
        elif v[0] == 1:
            row = [kernel_name, instructions, v[1][0], v[1][1]]
        writer.writerow(row)
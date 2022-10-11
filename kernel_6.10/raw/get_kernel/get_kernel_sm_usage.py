# 拟合 GPU的SM利用率
import csv
import collections
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


models = ["resnet50", "vgg19", "densenet201", "mobilenet"]
batch = [1, 8, 16, 32]
gpu10_kernel_sm_usage_hash = collections.defaultdict(list)
gpu25_kernel_sm_usage_hash = collections.defaultdict(list)
gpu50_kernel_sm_usage_hash = collections.defaultdict(list)
gpu75_kernel_sm_usage_hash = collections.defaultdict(list)
gpu100_kernel_sm_usage_hash = collections.defaultdict(list)

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
                    gpu10_kernel_sm_usage_hash[kernel_type].append(row[14])
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
                    gpu25_kernel_sm_usage_hash[kernel_type].append(row[14])

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
                    gpu50_kernel_sm_usage_hash[kernel_type].append(row[14])

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
                    gpu75_kernel_sm_usage_hash[kernel_type].append(row[14])

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
                    gpu100_kernel_sm_usage_hash[kernel_type].append(row[14])



# 对GPU资源为10时，各类型kernel的指令执行速度进行统计，并对多次出现的kernel速度进行平均
for k, v in gpu10_kernel_sm_usage_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu10_kernel_sm_usage_hash[k] = average_speed

for k, v in gpu25_kernel_sm_usage_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu25_kernel_sm_usage_hash[k] = average_speed

for k, v in gpu50_kernel_sm_usage_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu50_kernel_sm_usage_hash[k] = average_speed

for k, v in gpu75_kernel_sm_usage_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu75_kernel_sm_usage_hash[k] = average_speed

for k, v in gpu100_kernel_sm_usage_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu100_kernel_sm_usage_hash[k] = average_speed


# 统计所有kernel的类型
total_kernel_types = collections.defaultdict(lambda:[-1,-1,-1,-1,-1])
for k, v in gpu10_kernel_sm_usage_hash.items():
    origin = total_kernel_types[k]
    origin[0] = v
    total_kernel_types[k] = origin

for k, v in gpu25_kernel_sm_usage_hash.items():
    origin = total_kernel_types[k]
    origin[1] = v
    total_kernel_types[k] = origin

for k, v in gpu50_kernel_sm_usage_hash.items():
    origin = total_kernel_types[k]
    origin[2] = v
    total_kernel_types[k] = origin

for k, v in gpu75_kernel_sm_usage_hash.items():
    origin = total_kernel_types[k]
    origin[3] = v
    total_kernel_types[k] = origin

for k, v in gpu100_kernel_sm_usage_hash.items():
    origin = total_kernel_types[k]
    origin[4] = v
    total_kernel_types[k] = origin


def func2(x, k, b1, b2):
    r = k / (x + b1) + b2
    return r

def func1(x, k, b):
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
        popt, pcov = curve_fit(func2, x_data, y_data, maxfev=300000)
        kernel_kx_hash[k] = (1, popt)
    else:
        kernel_kx_hash[k] = (0, sum(y_data) / len(y_data))

res_file = "../predict/kernel_sm_usage.csv"
with open(res_file, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel_name", "instructions", "sm_usage_k", "sm_usage_b1", "sm_usage_b2", "no_kx_onlyC"])

    for k, v in sorted(kernel_kx_hash.items()):
        kernel_name = k[0]
        instructions = k[1]
        row = []
        if v[0] == 0:
            row = [kernel_name, instructions, "", "", "", v[1]]
        elif v[0] == 1:
            row = [kernel_name, instructions, v[1][0], v[1][1], v[1][2]]
        writer.writerow(row)

# with open(res_file, mode="w", encoding="utf-8-sig", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["kernel_name", "instructions", "sm_usage_k", "sm_usage_b", "no_kx_onlyC"])
#
#     for k, v in sorted(kernel_kx_hash.items()):
#         kernel_name = k[0]
#         instructions = k[1]
#         row = []
#         if v[0] == 0:
#             row = [kernel_name, instructions, "", "", v[1]]
#         elif v[0] == 1:
#             row = [kernel_name, instructions, v[1][0], v[1][1]]
#         writer.writerow(row)

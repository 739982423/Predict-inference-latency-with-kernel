import csv
import collections
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

models = ["resnet50", "vgg19", "densenet201", "mobilenet"]
batch = [1, 8, 16, 32]

gpu10_gpu_speed_hash = collections.defaultdict(list)
gpu25_gpu_speed_hash = collections.defaultdict(list)
gpu50_gpu_speed_hash = collections.defaultdict(list)
gpu75_gpu_speed_hash = collections.defaultdict(list)
gpu100_gpu_speed_hash = collections.defaultdict(list)

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
                    if row[11] == "":
                        print(file_name, row[11])
                    gpu10_gpu_speed_hash[kernel_type].append(row[11])
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
                    gpu25_gpu_speed_hash[kernel_type].append(row[11])

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
                    gpu50_gpu_speed_hash[kernel_type].append(row[11])

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
                    gpu75_gpu_speed_hash[kernel_type].append(row[11])

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
                    gpu100_gpu_speed_hash[kernel_type].append(row[11])

# 对GPU资源为10时，各类型kernel的指令执行速度进行统计，并对多次出现的kernel速度进行平均
for k, v in gpu10_gpu_speed_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu10_gpu_speed_hash[k] = average_speed

for k, v in gpu25_gpu_speed_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu25_gpu_speed_hash[k] = average_speed

for k, v in gpu50_gpu_speed_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu50_gpu_speed_hash[k] = average_speed

for k, v in gpu75_gpu_speed_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu75_gpu_speed_hash[k] = average_speed

for k, v in gpu100_gpu_speed_hash.items():
    new_speed = 0
    for num in v:
        new_speed += float(num)
    average_speed = new_speed / len(v)
    gpu100_gpu_speed_hash[k] = average_speed


# 统计所有kernel的类型
total_kernel_types = collections.defaultdict(lambda:[-1,-1,-1,-1,-1])
for k, v in gpu10_gpu_speed_hash.items():
    origin = total_kernel_types[k]
    origin[0] = v
    total_kernel_types[k] = origin

for k, v in gpu25_gpu_speed_hash.items():
    origin = total_kernel_types[k]
    origin[1] = v
    total_kernel_types[k] = origin

for k, v in gpu50_gpu_speed_hash.items():
    origin = total_kernel_types[k]
    origin[2] = v
    total_kernel_types[k] = origin

for k, v in gpu75_gpu_speed_hash.items():
    origin = total_kernel_types[k]
    origin[3] = v
    total_kernel_types[k] = origin

for k, v in gpu100_gpu_speed_hash.items():
    origin = total_kernel_types[k]
    origin[4] = v
    total_kernel_types[k] = origin

# for k, v in total_kernel_types.items():
#     print('------------')
#     print(k)
#     print(v)

def func1(x, k, b):
    r = k * x + b
    return r


kernel_kx_hash = collections.defaultdict(list)
for k, v in total_kernel_types.items():
    print('------------')
    print(k)
    print(v)
    candidate_x_data = [10, 25, 50, 75, 100]
    # candidate_x_data = [10, 50, 100]
    x_data = []
    y_data = []
    for idx, val in enumerate(v):
        if val != -1:
            x_data.append(candidate_x_data[idx])
            y_data.append(val)
    if len(x_data) >= 2:
        popt, pcov = curve_fit(func1, x_data, y_data)
        kernel_kx_hash[k] = (1, popt)
    else:
        kernel_kx_hash[k] = (0, y_data[0])

res_file = "../predict/kernel_gpu_speed.csv"
with open(res_file, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel_name", "instructions", "k_sm_frequency", "b_sm_frequency","no_kx_onlyC"])

    for k, v in sorted(kernel_kx_hash.items()):
        kernel_name = k[0]
        instructions = k[1]
        row = []
        if v[0] == 0:
            row = [kernel_name, instructions, "", "", v[1]]
        elif v[0] == 1:
            row = [kernel_name, instructions, v[1][0], v[1][1]]
        writer.writerow(row)

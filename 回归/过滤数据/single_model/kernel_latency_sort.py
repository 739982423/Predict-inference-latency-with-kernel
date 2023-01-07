# 该文件是在get_kernel_features.py使用产生kernel_summary.csv之后
# 目的是将每个类型的kernel平均时延排序，从小到大为每种类型的kernel进行编号
# 以便可以将种类作为一维数据通入决策树(回归树)

import os
import csv
import collections

summary_file = "kernel_summary.csv"
with open(summary_file, mode="r", encoding="gbk") as f:
    reader = csv.reader(f)
    kernel_data_hash = collections.defaultdict(list)
    line = 0
    for row in reader:
        if line < 2:
            line += 1
            continue
        # 从第一个ID可以转换成数字的行开始，因为summary的前两行是tips和title
        ID = int(row[0])
        kernel_name = row[2]        # 核函数名称
        kernel_latency = float(row[7]) # 核函数时延
        # 将 每个核函数的时延 放进 对应种类的哈希表里
        kernel_data_hash[kernel_name].append(kernel_latency)

# 定义每个类型对应的平均时延的存储结构，这里使用tuple+list的形式，方便排序
type_avglatency_list = []

# 对存储了每个类型所有核函数时延的哈希表进行遍历，求每个类型核函数的平均时延
for k, v in kernel_data_hash.items():
    # 用(核函数名称，平均时延)这样的元组来保存结果
    pair = (k, sum(v) / len(v))
    type_avglatency_list.append(pair)

# 对结果进行排序(以平均时延作为排序条件)
type_avglatency_list.sort(key = lambda x: x[1])

# 输出排序后的结果，因为核函数名称很长，所以只输出了排序后的平均时延
for name, latency in type_avglatency_list:
    print(latency)

# 最后存储每个类型对应的编号
result_file = "kernel_idx.csv"
with open(result_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    # 先写入一个title
    writer.writerow(["kernel name", "id", "avg latency(us)"])
    # 编号从1开始，表示平均时延最短的核函数
    idx = 1
    for name, latency in type_avglatency_list:
        row = [name, idx, latency]
        writer.writerow(row)
        idx += 1

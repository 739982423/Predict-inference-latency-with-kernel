# 本文件将使用决策树对核函数进行分类(预测)
# 需要提前准备好的有核函数的信息汇总表格kernel_summary.csv, 核函数编号信息表kernel_idx.csv

import collections
import csv
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

def number_filter(str):
    res = ""
    for s in str:
        if s not in "0123456789" and s != ".":
            continue
        else:
            res += s
    return float(res)

# 读取kernel信息
kernel_name_data_hash = collections.defaultdict(list)

total_input = collections.defaultdict(list)
total_target = collections.defaultdict(list)

summary_file = "kernel_summary.csv"
with open(summary_file, mode="r", encoding="gbk") as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line < 2:
            line += 1
            continue

        ID = float(row[0])
        # 拿出有用的信息，这里筛选出要作为决策树输入的特征
        kernel_name = row[2]
        kernel_name_data_hash[row[2]].append(row)
        # 这里是kernel对应的data部分
        instructions = number_filter(row[10])
        lanuch_block = number_filter(row[12])
        lanuch_grid = number_filter(row[13])
        sm_usage = number_filter(row[14])
        # l1向l2读和写的数量
        l1_read = number_filter(row[22])
        l1_write = number_filter(row[23])
        # l2向memory读和写的数量
        l2_read = number_filter(row[32])
        l2_write = number_filter(row[33])

        # 生成输入数据，这里不需要kernel的id了，因为每个大类kernel自行使用一个决策树
        kernel_data = [instructions, lanuch_block, lanuch_grid, sm_usage, l1_read, l1_write, l2_read, l2_write]
        kernel_latency = number_filter(row[7])

        # 保存在总体list中
        total_input[kernel_name].append(kernel_data)
        total_target[kernel_name].append(kernel_latency)

# 展示每个类型kernel各有几个，大部分超过100个，只有3种是两个，可以将这3种忽略
for k, v in kernel_name_data_hash.items():
    if len(v) < 10:
        continue
    kernel_name = k
    cur_input = total_input[kernel_name]
    cur_target = total_target[kernel_name]
    print(cur_input)
    print(len(cur_input))
    print(cur_target)
    print(len(cur_target))
    # x_train, x_test, y_train, y_test = train_test_split(cur_input, cur_target, test_size=0.1)
    # dtr = DecisionTreeRegressor(max_depth=5)
    # dtr.fit(x_train, y_train)
    # dtr_y_predict = dtr.predict(x_test)
    # for i in range(len(dtr_y_predict)):
    #     # print(dtr_y_predict[i], x_test[i])
    # # print("----------------")
    # # print(len(cur_input))
    # print('R-squared value of DecisionTreeRegressor:', dtr.score(x_test, y_test))
    # # print('The mean squared error of DecisionTreeRegressor:', mean_squared_error(y_test, dtr_y_predict))
    # # print('The mean absolute error of DecisionTreeRegressor:', mean_absolute_error(y_test, dtr_y_predict))
    break






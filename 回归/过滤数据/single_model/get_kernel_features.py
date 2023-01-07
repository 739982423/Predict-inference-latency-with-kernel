# 因为经过kernel种类的统计，已知有35873个kernel，有49种类型
# 因此，在本文件，我们开始整理这些kernel的特征，并将所有kernel的信息按分类排序，输出到一个excel中
# 输出文件为当前目录下的kernel_summary.csv

import os
import csv
import collections

dir = os.listdir("./filtered_data")

total_type_num_hash = collections.defaultdict(list)
tips = ["tips", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "L1/TEX 读取L2 CACHE的数据量(mb)",
         "L1/TEX 读取L2 CACHE数据的速度(gb/s)","L1/TEX 写入L2 CACHE的数据量(mb)","L1/TEX 写入L2 CACHE数据的速度(gb/s)",
         "L1/TEX LOAD requests","L1/TEX STORE requests","L1/TEX LOAD requests + L1/TEX STORE requests",
         "L1/TEX LOAD sectors(L1向L2读的数据量)","L1/TEX STORE sectors(L1向L2写的数据量)",
         "lts__t_sectors_lookup_hit.sum","lts__t_sectors_lookup_miss.sum 这俩是计算Hit Rate的",
         "L1/TEX LOAD Hit sectors sum","L1/TEX LOAD Miss sectors sum","Device memory LOAD sectors",
         "Device memory STORE sectors","Percentage of peak device memory utilization(device memory loads)",
         "Percentage of peak device memory utilization(device memory stores)",
         "l2从device dram读的数据量(sector)", "l2向device dram写的数据量(sector)"
        ]
title = []
for file_name in dir:
    if file_name[-4:] == ".csv" and "kernel" not in file_name:
        print(file_name)
    else:
        break

    with open("./filtered_data/" + file_name, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        i = 0
        m_second = False
        for row in reader:
            if title == []:
                if row[7] == "gpu__time_duration.sum [msecond]":
                    row[7] = "gpu__time_duration.sum [usecond]"
                title = row
            if i == 0:
                # 固定了GPU时间单位为us
                # 如果该文件内的时延是以ms为单位，则该文件内所有的时延×1000
                if row[7] == "gpu__time_duration.sum [msecond]":
                    m_second = True
                i += 1
                continue
            if m_second:
                row[7] = str(float(row[7]) * 1000)
            total_type_num_hash[row[2]].append(row)

    # 输出最终统计了多少种kernel，结果为49说明统计完全
    # 输出最终统计了多少个kernel，结果为35873说明统计完全
    types_cnt = 0
    nums_cnt = 0
    for k, v in total_type_num_hash.items():
        types_cnt += 1
        nums_cnt += len(v)
print("当前哈希表内有{}种kernel，总数为{}".format(types_cnt, nums_cnt))

# 接下来将kernel信息存入一个表
total_file_name = "kernel_summary.csv"
with open(total_file_name, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(tips)
    writer.writerow(title)
    for type, kernel_list in total_type_num_hash.items():
        for kernel in kernel_list:
            writer.writerow(kernel)

# 该文件用来统计每个文件内有多少个、有多少种kernel
# 全体80个文件中，一共有多少个、多少种kernel

# 结果:
#     80个文件中，共有35873个kernel，种类为49种

import os
import csv
import collections

dir = os.listdir("./filtered_data")

# loop = 4
total_nums = 0
total_types = 0
total_type_num_hash = collections.defaultdict(int)
for file_name in dir:
    if file_name[-4:] == ".csv":
        print(file_name)
    else:
        continue
    kernel_nums = 0
    kernel_types = 0
    type_num_hash = collections.defaultdict(int)
    with open("./filtered_data/" + file_name, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            # 首行标题跳过
            if i == 0:
                i += 1
                continue

            if type_num_hash[row[2]] == 0:
                kernel_types += 1
            type_num_hash[row[2]] += 1
            kernel_nums += 1

            if total_type_num_hash[row[2]] == 0:
                total_types += 1
            total_type_num_hash[row[2]] += 1
            total_nums += 1
    print("kernel nums =", kernel_nums)
    print("kernel types =", kernel_types)
    # for k, v in type_num_hash.items():
    #     print(k[:20], v)
    print("-------------------------------------------------------------")

    # loop -= 1
    # if loop == 0:
    #     break

print("统计完毕:")
print("total kernel nums =", total_nums)
print("total kernel types =", total_types)

# 验证CUDA API调用与实际执行的kernel之间是否一对一对应
# 结论，否。相同API调用可能会执行不同kernel。
# 但初步验证相同API实际需要执行的计算量相同（即使实际执行的kernel不同）

import csv
from collections import defaultdict
csv_name = ["resnet50_b1_g100_namesorted.csv", "resnet50_b8_g100_namesorted.csv",
            "vgg19_b1_g100_namesorted.csv", "vgg19_b8_g100_namesorted.csv"]

function_conut = defaultdict(int)
kernel_count = defaultdict(int)
f_k_hash = defaultdict(str)

pass_count = 0
error_count = 0
total_count = 0
for csv_file in csv_name:
    with open(csv_file, mode = "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "ID":
                continue
            total_count += 1
            cur_func_name = row[2]
            cur_kernel_name = row[3]
            function_conut[cur_func_name] += 1
            kernel_count[cur_kernel_name] += 1
            if f_k_hash[cur_func_name] == "":
                f_k_hash[cur_func_name] = cur_kernel_name
                pass_count += 1
            else:
                if f_k_hash[cur_func_name] == cur_kernel_name:
                    pass_count += 1
                    print("验证通过")
                else:
                    error_count += 1
                    print("{}: id = {}的函数验证失败".format(csv_file, row[0]))
                    print("function name:{}".format(cur_func_name))
                    print("kernel name:{}".format(cur_kernel_name))
print("--------------END--------------")
print("共计{}个(func, kernel)对，验证通过{}对，失败{}对".format(total_count, pass_count, error_count))
#该文件测试不同网络在不同GPU资源条件和batch下latency和GPU资源的关系
import collections
import csv

model_name = ["vgg19"]
batch = ["1"]
gpu_resource = ["10", "50", "100"]
gpu10_kernel_hash = collections.defaultdict(list)
gpu50_kernel_hash = collections.defaultdict(list)
gpu100_kernel_hash = collections.defaultdict(list)

title = []
for model in model_name:
    for b in batch:
        for resource in gpu_resource:
            file_name = "filtered_raw_" + model + "_b" + b + "_g" + resource + ".csv"
            print(file_name)
            with open(file_name, mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)

                # 参数位置
                idx = 0
                function_name = 1
                kernel_name = 2
                gpu_cycles = 6
                gpu_time = 7
                active_cycles = 8
                inst_speed = 9
                instructions = 10

                data = []
                for row in reader:

                    if resource == "10":
                        if row[0] == "ID":
                            title = row
                            continue
                        else:
                            data = row
                        kernel_type = (data[2], data[10])
                        gpu10_kernel_hash[kernel_type].append(data)
                    if resource == "50":
                        if row[0] == "ID":
                            continue
                        else:
                            data = row
                        kernel_type = (data[2], data[10])
                        gpu50_kernel_hash[kernel_type].append(data)
                    if resource == "100":
                        if row[0] == "ID":
                            continue
                        else:
                            data = row
                        kernel_type = (data[2], data[10])
                        gpu100_kernel_hash[kernel_type].append(data)

# for k, v in gpu100_kernel_hash.items():
#     print("---")
#     print(k)
#     print(len(v))
#     print(v)

for k, v in gpu10_kernel_hash.items():
    resfile_name = "vgg19_b1_compare.csv"
    with open(resfile_name, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        write_title = ["", title[2], title[10], "", title[6], title[7], title[8], title[9]]

        for i in range(len(v)):
            writer.writerow(write_title)
            for gpu_resource_type in range(3):
                if gpu_resource_type == 0:
                    data_resource_hash = gpu10_kernel_hash
                    gpu_resource_percent = 10
                elif gpu_resource_type == 1:
                    data_resource_hash = gpu50_kernel_hash
                    gpu_resource_percent = 50
                elif gpu_resource_type == 2:
                    data_resource_hash = gpu100_kernel_hash
                    gpu_resource_percent = 100
                # print("-------")
                # print(k)
                # print(data_resource_hash[k])
                # print(len(data_resource_hash[k]))
                cur_data = data_resource_hash[k][0]
                write_line = ["gpu" + str(gpu_resource_percent), cur_data[2], cur_data[10],
                              "", cur_data[6], cur_data[7], cur_data[8], cur_data[9]]
                writer.writerow(write_line)
                writer.writerow([])


import csv
import collections

m1_name = "densenet201"
m2_name = "mobilenet"
m3_name = "resnet50"
m4_name = "vgg19"

origin_csv_name = m1_name[0] + "_" + m2_name[0] + "_" + m3_name[0] + "_" + m4_name[0] + ".csv"
# target_csv_name = "filtered2_" + origin_csv_name
target_csv_name = "tmp_" + origin_csv_name

m1_origin_csv = "./../base/4_models_base/" + m1_name[0] + ".csv"
m2_origin_csv = "./../base/4_models_base/" + m2_name[0] + ".csv"
m3_origin_csv = "./../base/4_models_base/" + m3_name[0] + ".csv"
m4_origin_csv = "./../base/4_models_base/" + m4_name[0] + ".csv"

m1_origin_latency = []
m2_origin_latency = []
m3_origin_latency = []
m4_origin_latency = []

memory_csv = "C:\\Users\\73998\Desktop\实验数据\kernel_6.10\\raw\get_kernel\hit_rate_message.csv"

memory_message_hash = collections.defaultdict(list)
# memory_message_hash['title'] = ["hit sum","miss sum","hit+miss sum","hit rate","l1 read from l2","l1 write to l2","l1 read+write from/to l2","l2 read from dram","l2 write to dram", "l2 read+write from/to dram"]

with open(memory_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if len(row) > 0:
            bs = row[1]
            gpu = row[2]
            message = [row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]]
            memory_message_hash[row[0] + "_b" + row[1] + "_g" + row[2]] = message

# 读取第一个模型的共存信息
with open(m1_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m1_origin_latency.append(row)

# 读取第二个模型的共存信息
with open(m2_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m2_origin_latency.append(row)

# 读取第三个模型的共存信息
with open(m3_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m3_origin_latency.append(row)


# 读取第四个模型的共存信息
with open(m4_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m4_origin_latency.append(row)

res_row = []
# 读取原始数据
with open(origin_csv_name, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != "":
            res_row.append(row)
# print(res_row)
# ---------------------------------------------------------------------------------------------------

bs = [1, 8, 16]
gpu_resource = [10, 25, 50, 75]

print(m4_origin_latency)
for g1 in gpu_resource:
    for b1 in bs:
        #模型1某个配置的变化趋势的结果
        tmp_res = []
        idx = 0
        n = len(res_row)
        for i in range(len(res_row) - 4):
            if res_row[i][0] == "model name" and res_row[i + 1][0] == "densenet201" and res_row[i + 2][0] == "mobilenet"\
                    and res_row[i + 3][0] == "resnet50" and res_row[i + 4][0] == "vgg19" and res_row[i + 1][1] == str(g1)\
                    and res_row[i + 1][2] == str(b1):
                # 第一个模型是densenet201
                m1_g = g1
                m1_bs = b1
                single_latency = float(
                    m1_origin_latency[gpu_resource.index(m1_g)][bs.index(m1_bs)])
                # 一共七个属性：模型名称，GPU，bs，inference time，exec times，共存增长的时间，共存增长时间的比例
                m1_res = [res_row[i + 1][0], res_row[i + 1][1], res_row[i + 1][2], res_row[i + 1][5], res_row[i + 1][4],
                          float(res_row[i + 1][5]) - single_latency, (float(res_row[i + 1][5]) - single_latency) / single_latency]


                # 下一个模型是mobilenet
                m2_g = int(res_row[i + 2][1])
                m2_bs = int(res_row[i + 2][2])
                single_latency = float(
                    m2_origin_latency[gpu_resource.index(m2_g)][bs.index(m2_bs)])
                # 一共七个属性：模型名称，GPU，bs，inference time，exec times，共存增长的时间，共存增长时间的比例
                m2_res = [res_row[i + 2][0], res_row[i + 2][1], res_row[i + 2][2], res_row[i + 2][5], res_row[i + 2][4],
                          float(res_row[i + 2][5]) - single_latency, (float(res_row[i + 2][5]) - single_latency) / single_latency]

                # 下一个模型是resnet50
                m3_g = int(res_row[i + 3][1])
                m3_bs = int(res_row[i + 3][2])
                single_latency = float(
                    m3_origin_latency[gpu_resource.index(m3_g)][bs.index(m3_bs)])
                # 一共七个属性：模型名称，GPU，bs，inference time，exec times，共存增长的时间，共存增长时间的比例
                m3_res = [res_row[i + 3][0], res_row[i + 3][1], res_row[i + 3][2], res_row[i + 3][5], res_row[i + 3][4],
                          float(res_row[i + 3][5]) - single_latency, (float(res_row[i + 3][5]) - single_latency) / single_latency]


                # 最后一个模型是vgg19
                m4_g = int(res_row[i + 4][1])
                m4_bs = int(res_row[i + 4][2])
                single_latency = float(
                    m4_origin_latency[gpu_resource.index(m4_g)][bs.index(m4_bs)])
                # 一共七个属性：模型名称，GPU，bs，inference time，exec times，共存增长的时间，共存增长时间的比例
                m4_res = [res_row[i + 4][0], res_row[i + 4][1], res_row[i + 4][2], res_row[i + 4][5], res_row[i + 4][4],
                          float(res_row[i + 4][5]) - single_latency, (float(res_row[i + 4][5]) - single_latency) / single_latency]


                tmp_res.append((m1_res, m2_res, m3_res, m4_res))

        with open(target_csv_name, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)

            for i in range(len(tmp_res)):
                cur_row = tmp_res[i][0] + [""] + tmp_res[i][1] + [""] + tmp_res[i][2] + [""] + tmp_res[i][3]

                writer.writerow(cur_row)
import csv
import collections

m1_name = "densenet201"
m2_name = "vgg19"
origin_csv_name = m1_name[0] + "_" + m2_name[0] + ".csv"
# target_csv_name = "filtered2_" + origin_csv_name
target_csv_name = "tmp_" + origin_csv_name

m1_origin_csv = "./../base/" + m1_name[0] + ".csv"
m2_origin_csv = "./../base/" + m2_name[0] + ".csv"
m1_origin_latency = []
m2_origin_latency = []

memory_csv = "C:\\Users\\73998\Desktop\实验数据\kernel_6.10\\raw\get_kernel\hit_rate_message.csv"

memory_message_hash = collections.defaultdict(list)
memory_message_hash['title'] = ["hit sum","miss sum","hit rate","l1 read from l2","l1 write to l2","l2 read from dram","l2 write to dram"]

with open(memory_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 0 and row[0] == "vgg19":
            bs = row[1]
            gpu = row[2]
            message = [row[3], row[4], row[5], row[6], row[7], row[8], row[9]]
            memory_message_hash[row[0] + "_b" + row[1] + "_g" + row[2]] = message
print(memory_message_hash)
with open(m1_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m1_origin_latency.append(row)

with open(m2_origin_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        m2_origin_latency.append(row)

bs = [1, 8, 16, 32]
gpu_resource = [10, 25, 50, 75, 90]

res_row = []

with open(origin_csv_name, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        res_row.append(row)
# print(res_row)

for g1 in gpu_resource:
    for b1 in bs:
        #模型1某个配置的变化趋势的结果
        tmp_res = []
        for idx, row in enumerate(res_row):
            if len(row) < 6:
                continue
            # print(row)
            if row[0] == m1_name and row[1] == str(g1) and row[2] == str(b1):
                m1_res = [row[0], row[1], row[2], row[5], row[4]]
                m2_res = []
                if res_row[idx - 1][0] == m2_name:
                    m2_res = [res_row[idx - 1][0], res_row[idx - 1][1], res_row[idx - 1][2], res_row[idx - 1][5], res_row[idx - 1][4]]
                elif idx + 1 < len(res_row) and res_row[idx + 1][0] == m2_name:
                    m2_res = [res_row[idx + 1][0], res_row[idx + 1][1], res_row[idx + 1][2], res_row[idx + 1][5], res_row[idx + 1][4]]
                tmp_res.append((m1_res, m2_res))
        with open(target_csv_name, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            single_latency = m1_origin_latency[gpu_resource.index(g1)][bs.index(b1)]
            writer.writerow([m1_name, "origin", "", single_latency])
            for pair in tmp_res:
                pair[0].append(float(pair[0][3]) - float(single_latency))
                pair[0].append((float(pair[0][3])  - float(single_latency)) / float(single_latency))
                pair[0].append("")
                if len(pair[1]) >= 3:
                    m2_g = int(pair[1][1])
                    m2_bs = int(pair[1][2])
                    m2_single_latency = m2_origin_latency[gpu_resource.index(m2_g)][bs.index(m2_bs)]
                    pair[1].append(float(pair[1][3]) - float(m2_single_latency))
                    pair[1].append((float(pair[1][3]) - float(m2_single_latency)) / float(m2_single_latency))
                    pair[1].append("")

                    # 之后是m2的memory相关的数据部分
                    key_name = m2_name + "_b" + str(m2_bs) + "_g" + str(m2_g)
                    extra_data = memory_message_hash[key_name]

                writer.writerow(pair[0] + pair[1] + extra_data)
            writer.writerow([])

for g1 in gpu_resource:
    for b1 in bs:
        #模型1某个配置的变化趋势的结果
        tmp_res = []
        for idx, row in enumerate(res_row):
            if len(row) < 6:
                continue
            if row[0] == m2_name and row[1] == str(g1) and row[2] == str(b1):
                m1_res = [row[0], row[1], row[2], row[5], row[4]]
                m2_res = []
                if res_row[idx - 1][0] == m1_name:
                    m2_res = [res_row[idx - 1][0], res_row[idx - 1][1], res_row[idx - 1][2], res_row[idx - 1][5], res_row[idx - 1][4]]
                elif idx + 1 < len(res_row) and res_row[idx + 1][0] == m1_name:
                    m2_res = [res_row[idx + 1][0], res_row[idx + 1][1], res_row[idx + 1][2], res_row[idx + 1][5], res_row[idx + 1][4]]
                tmp_res.append((m1_res, m2_res))
        with open(target_csv_name, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            single_latency = m2_origin_latency[gpu_resource.index(g1)][bs.index(b1)]
            writer.writerow([m2_name, "origin", "", single_latency])
            for pair in tmp_res:
                pair[0].append(float(pair[0][3])  - float(single_latency))
                pair[0].append((float(pair[0][3])  - float(single_latency)) / float(single_latency))
                pair[0].append("")
                if len(pair[1]) >= 3:
                    m1_g = int(pair[1][1])
                    m1_bs = int(pair[1][2])
                    m1_single_latency = m1_origin_latency[gpu_resource.index(m1_g)][bs.index(m1_bs)]
                    pair[1].append(float(pair[1][3]) - float(m1_single_latency))
                    pair[1].append((float(pair[1][3]) - float(m1_single_latency)) / float(m1_single_latency))

                    # 之后是m1的memory相关的数据部分
                    key_name = m1_name + "_b" + str(m1_bs) + "_g" + str(m1_g)
                    extra_data = memory_message_hash[key_name]
                writer.writerow(pair[0] + pair[1] + extra_data)
            writer.writerow([])
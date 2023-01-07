# 该文件是用来对两个模型共存情况的数据进行预处理的
# 主要预处理的部分是，对每一次两模型共存的运行，整理出两模型在单位时间内的多个与L2相关的数据交换量
# 该文件需要读取2个模型的共存执行数据，即tmp_d_v.csv, tmp_r_m.csv等文件，将输出res_tmp_d_v.csv等文件作为结果
# res_tmp_d_v.csv文件中会在每一行共存执行结果后面附加每个模型执行过程中与L2相关的四个数据交换量的归一化值

import collections
import csv

data_file = ["tmp_d_m.csv", "tmp_d_v.csv", "tmp_r_d.csv", "tmp_r_m.csv", "tmp_r_v.csv", "tmp_v_m.csv"]
summary_file = "summary.csv"

data_transfer_summary = collections.defaultdict(list)
processed_line = list()

with open(summary_file, mode="r", encoding="gbk") as f:
    reader = csv.reader(f)
    # 跳过summary文件的标题两行
    line = 0
    for row in reader:
        if line < 2:
            line += 1
            continue
        model_name = row[0]
        bs = row[1]
        gpu = row[2]

        # 由模型名称+batch+gpu资源量组成哈希表的key，以key存储其data transfer的数据
        key = model_name + "_" + bs + "_" + gpu
        data_transfer_summary[key] = row

for file in data_file:
    with open(file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == "origin" or row[1] == "" or row[8] == "":
                continue
            # 分别读取两个模型的数据，latency是受影响情况下的时延，increase是相比与单独运行增长的时延(毫秒，非百分比)，increase percent是增长的百分比
            m1_name, m2_name = row[0], row[8]
            m1_gpu, m2_gpu = row[1], row[9]
            if m1_gpu == "90" or m2_gpu == "90":
                continue
            m1_bs, m2_bs = row[2], row[10]
            m1_latency , m2_latency = row[3], row[11]
            m1_increase, m2_increase = row[5], row[13]
            m1_increase_percent, m2_increase_precent = row[6], row[14]
            # 以上数据都是字符串类型

            # 接下来开始计算每个模型单位时间内，四项与L2相关指标的数据交换量
            # 首先读取两个模型与L2相关的四项指标
            key1 = m1_name + "_" + m1_bs + "_" + m1_gpu
            key2 = m2_name + "_" + m2_bs + "_" + m2_gpu

            # ----------------------处理第一个模型的部分-----------------------
            summary_line = data_transfer_summary[key1]
            # print(summary_line)
            try:
                l1_read_from_l2 = summary_line[7]
                l1_write_to_l2 = summary_line[8]
                l2_read_from_dram = summary_line[10]
                l2_write_to_dram = summary_line[11]
            except:
                print("m1")
                print(key1, summary_line)
            # print(l1_read_from_l2, l1_write_to_l2)
            # print(l2_read_from_dram, l2_write_to_dram)
            normalized_l1_read_from_l2 = float(l1_read_from_l2) / float(m1_latency)
            normalized_l1_write_to_l2 = float(l1_write_to_l2) / float(m1_latency)
            normalized_l2_read_from_dram = float(l2_read_from_dram) / float(m1_latency)
            normalized_l2_write_to_dram = float(l2_write_to_dram) / float(m1_latency)
            # print(normalized_l1_read_from_l2, normalized_l1_write_to_l2)
            # print(normalized_l2_read_from_dram, normalized_l2_write_to_dram)
            # 存储四个归一化结果
            L1_results =  [normalized_l1_read_from_l2, normalized_l1_write_to_l2, normalized_l2_read_from_dram,
                           normalized_l2_write_to_dram]

            # ----------------------处理第二个模型的部分-----------------------
            summary_line = data_transfer_summary[key2]
            try:
                l1_read_from_l2 = summary_line[7]
                l1_write_to_l2 = summary_line[8]
                l2_read_from_dram = summary_line[10]
                l2_write_to_dram = summary_line[11]
            except:
                print("m2")
                print(key2, summary_line, row)
            # print(l1_read_from_l2, l1_write_to_l2)
            # print(l2_read_from_dram, l2_write_to_dram)
            normalized_l1_read_from_l2 = float(l1_read_from_l2) / float(m1_latency)
            normalized_l1_write_to_l2 = float(l1_write_to_l2) / float(m1_latency)
            normalized_l2_read_from_dram = float(l2_read_from_dram) / float(m1_latency)
            normalized_l2_write_to_dram = float(l2_write_to_dram) / float(m1_latency)
            # print(normalized_l1_read_from_l2, normalized_l1_write_to_l2)
            # print(normalized_l2_read_from_dram, normalized_l2_write_to_dram)
            # 存储四个归一化结果
            L2_results =  [normalized_l1_read_from_l2, normalized_l1_write_to_l2, normalized_l2_read_from_dram,
                           normalized_l2_write_to_dram]

            #----------------------保存结果部分-----------------------
            result_row = row[:15]
            result_row.append(" ")
            result_row += L1_results
            result_row.append(" ")
            result_row += L2_results
            processed_line.append(result_row)
            # print(processed_line)
    result_file = "res_" + file
    with open(result_file, mode="w", encoding="gbk", newline="") as f:
        writer = csv.writer(f)
        for line in processed_line:
            writer.writerow(line)
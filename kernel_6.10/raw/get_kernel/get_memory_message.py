import csv

def get_number(s):
    s_list = s.split(",")
    res = 0
    cur_power = 0
    for i in range(len(s_list) - 1, -1, -1):
        res += float(s_list[i]) * pow(1000, cur_power)
        cur_power += 1
    return res

root_name = "filtered_raw_"
model_name = ["vgg19", "densenet201", "mobilenet", "resnet50"]
batch = [1, 8, 16, 32]
gpu_resource = [10, 25, 50, 75, 100]

res_csv_name = "hit_rate_message.csv"
with open(res_csv_name, mode="a", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    title = ["model name", "bs", "gpu resource", "hit sum", "miss sum", "hit+miss sum", "hit rate",
             "l1 read from l2", "l1 write to l2", "l1 read+write from/to l2",
             "l2 read from dram", "l2 write to dram", "l2 read+write from/to dram"]
    writer.writerow(title)

for m in model_name:
    for bs in batch:
        for gpu in gpu_resource:
            cur_file_name = root_name + m + "_b" + str(bs) + "_g" + str(gpu) + ".csv"
            hit_sum = 0
            miss_sum = 0
            sectors_l1_tex_read_from_l2 = 0
            sectors_l1_tex_write_to_l2 = 0
            sectors_l2_read_from_dram = 0
            sectors_l2_write_to_dram = 0
            with open(cur_file_name, mode = "r", encoding = "utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] != "ID" and row[0] != "":
                        hit_sum += get_number(row[26])
                        miss_sum += get_number(row[27])
                        sectors_l1_tex_read_from_l2 += get_number(row[22])
                        sectors_l1_tex_write_to_l2 += get_number(row[23])
                        sectors_l2_read_from_dram += get_number(row[32])
                        sectors_l2_write_to_dram += get_number(row[33])
            with open(res_csv_name, mode = "a", encoding = "utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                row = [m, bs, gpu, hit_sum, miss_sum, hit_sum + miss_sum, hit_sum / (hit_sum + miss_sum),
                       sectors_l1_tex_read_from_l2, sectors_l1_tex_write_to_l2,
                       sectors_l1_tex_read_from_l2 + sectors_l1_tex_write_to_l2,
                       sectors_l2_read_from_dram, sectors_l2_write_to_dram,
                       sectors_l2_read_from_dram + sectors_l2_write_to_dram]
                writer.writerow(row)
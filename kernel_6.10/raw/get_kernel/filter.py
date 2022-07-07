import csv
import os
cur_path = os.getcwd()
file_name_list = os.listdir(cur_path)

for file in file_name_list:
    if file[:3] != "raw":
        continue
    res_file_name = "filtered_" + file
    res = []
    with open(file, mode="r", encoding="utf-8-sig") as f:
        print(file)
        reader = csv.reader(f)
        for row in reader:
            tmp = []
            tmp.append(row[0])
            tmp.append(row[4])
            tmp.append(row[6])
            tmp.append(row[7])
            tmp.append(row[13])
            tmp.append(row[14])
            tmp.append(row[16])
            tmp.append(row[17])
            tmp.append(row[535])
            tmp.append(row[538])
            tmp.append(row[605])
            tmp.append(row[194])
            tmp.append(row[355])
            tmp.append(row[363])
            tmp.append(row[18])
            tmp.append(row[244])    # L1/TEX 读取L2 CACHE的数据量(mb)
            tmp.append(row[246])    # L1/TEX 读取L2 CACHE数据的速度(gb/s)
            tmp.append(row[229])    # L1/TEX 写入L2 CACHE的数据量(mb)
            tmp.append(row[231])    # L1/TEX 写入L2 CACHE数据的速度(gb/s)
            tmp.append(row[391])    # L1/TEX LOAD requests
            tmp.append(row[393])    # L1/TEX STORE requests
            tmp.append(row[387])    # L1/TEX LOAD requests + L1/TEX STORE requests
            tmp.append(row[464])    # L1/TEX LOAD sectors
            tmp.append(row[478])    # L1/TEX STORE sectors
            tmp.append(row[408])    # lts__t_sectors_lookup_hit.sum
            tmp.append(row[409])    # lts__t_sectors_lookup_miss.sum 这俩是计算Hit Rate的
            tmp.append(row[466])    # L1/TEX LOAD Hit sectors sum
            tmp.append(row[467])    # L1/TEX LOAD Miss sectors sum
            tmp.append(row[191])    # Device memory LOAD sectors
            tmp.append(row[192])    # Device memory STORE sectors
            tmp.append(row[184])    # Percentage of peak device memory utilization(device memory loads)
            tmp.append(row[187])    # Percentage of peak device memory utilization(device memory stores)
            tmp.append(row[191])    # l2从device dram load了多少sector
            tmp.append(row[192])    # l2向device dram write了多少sector
            res.append(tmp)

    with open(res_file_name, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for row in res:
            writer.writerow(row)


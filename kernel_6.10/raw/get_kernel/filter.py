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
            res.append(tmp)

    with open(res_file_name, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        for row in res:
            writer.writerow(row)


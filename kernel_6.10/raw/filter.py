import csv

origin_file_name = "raw_resnet50_b8_g50.csv"
res_file_name = "filtered_" + origin_file_name
res = []
with open(origin_file_name, mode="r", encoding="utf-8-sig") as f:
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

        res.append(tmp)

with open(res_file_name, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    for row in res:
        writer.writerow(row)


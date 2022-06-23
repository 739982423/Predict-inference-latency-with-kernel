import csv

model_name = "densenet201"
gpu_allocated = [10, 25, 50, 75, 100]
bs = [1, 8, 16, 32]


final_res = []
for jj in range(len(bs)):
    tmp_res = []
    for ii in range(len(gpu_allocated)):
        time_scale = 1
        profile_res_csv = "./kernel_data/filtered_raw_{}_b{}_g{}.csv".format(model_name, str(bs[jj]), str(gpu_allocated[ii]))
        total_time = 0
        with open(profile_res_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "ID":
                    if row[7] == "gpu__time_duration.sum [usecond]":
                        time_scale = 1000
                    else:
                        time_scale = 1
                else:
                    total_time += float(row[7])
        tmp_res.append(total_time / time_scale)
    final_res.append(tmp_res)

predict_res_csv = "profiled_duration.csv"
with open(predict_res_csv, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    lines = [[] for i in range(5)]
    for i in range(len(final_res[0])):
        lines[i].append(final_res[0][i])
        lines[i].append(final_res[1][i])
        lines[i].append(final_res[2][i])
        lines[i].append(final_res[3][i])


    for line in lines:
        writer.writerow(line)


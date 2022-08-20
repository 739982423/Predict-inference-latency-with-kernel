import collections
import csv
import os

root_dir = os.getcwd()
csv_name = "filtered_raw_densenet201_b1_g10"
model_name = ["densenet201"]
bs = [1, 8]
for model in model_name:
    cur_model_res = collections.defaultdict(dict)
    for batch in bs:
        tmp_res = []
        csv_name = "filtered_raw_" + model + "_b" + str(batch) + "_g100.csv"
        with open(csv_name, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                tmp_res.append(row)
        cur_model_res[batch] = tmp_res

        if batch == 8:
            pre_res_list = cur_model_res[1]
            cur_res_list = cur_model_res[8]
            pre_res_list.sort(key = lambda x: x[0])
            cur_res_list.sort(key = lambda x: x[0])
            diff_list = []
            i, j = 0, 0
            while(i < len(pre_res_list)):
                print(i, j)
                if pre_res_list[i][1] == cur_res_list[j][1]:
                    i += 1
                    j += 1
                else:
                    diff_list.append(cur_res_list[j][1])
                    j += 1
            for i in range(len(diff_list)):
                print(diff_list[i][1])


# 获取每个csv文件对应配置下所有kernel的执行时间总和（以此估算latency）
import csv
csv_name = ["resnet50_b1_g100_namesorted_2.csv", "resnet50_b8_g100_namesorted_2.csv",
            "vgg19_b1_g100_namesorted_2.csv", "vgg19_b8_g100_namesorted_2.csv"]

us_flag = True
for csv_file in csv_name:
    time = 0
    message = csv_file.split("_")
    model_name, bs, gpu_resource = message[0], message[1], message[2]
    with open(csv_file, mode = "r", encoding=" utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            # 获取excel内时间的单位
            # print(row)
            if row[0] == "ID":
                time_scale_name = row[9]
                if time_scale_name == "Duration [usecond]":
                    us_flag = True
                elif time_scale_name == "Duration [msecond]":
                    us_flag = False
            else:
                time += float(row[9])
    print("{}_{}_{} latency = {} {}".format(model_name, bs, gpu_resource, time, time_scale_name[-9:]))

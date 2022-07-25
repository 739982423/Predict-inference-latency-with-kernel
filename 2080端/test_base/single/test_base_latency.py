import os
import torch
from torchvision import models as models
import time
import csv

device = torch.device('cuda')
# model = models.mobilenet.mobilenet_v2().to(device)
model1 = models.resnet50().to(device)
model2 = models.vgg19().to(device)
model3 = models.densenet201().to(device)

model_list = [model1]
model_name = "mobilenet"



bs_list = [1, 4, 8, 16]
# gpu_resource_list = [10]
# bs_list = [4, 12]
gpu_resource_list = [10, 25, 50, 75]

res_matrix = [[0] * len(bs_list) for _ in range(len(gpu_resource_list))]

for i in range(len(gpu_resource_list)):
    os.system("echo quit | sudo nvidia-cuda-mps-control")    
    time.sleep(1)
    os.system("sudo nvidia-cuda-mps-control -d")  
    time.sleep(1)
    os.system("echo set_default_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(str(gpu_resource_list[i])))
    time.sleep(1)
    for j in range(len(bs_list)):
        print("python model_exec.py -m {} -b {} -g{}".format(model_name, bs_list[j], gpu_resource_list[i]))
        os.system("python model_exec.py -m {} -b {} -g{}".format(model_name, bs_list[j], gpu_resource_list[i]))
        time.sleep(1)
        with open("tmp_res.csv", mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                res = row[0]
                res_matrix[i][j] = res

with open("final_res.csv", mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    for row in res_matrix:
        writer.writerow(row)

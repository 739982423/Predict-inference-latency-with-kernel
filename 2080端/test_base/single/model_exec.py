import time
import torch
from torchvision import models as models
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="tested model")
parser.add_argument("-b", "--batch", help="batch", type=int)
parser.add_argument("-g", "--gpu", help="gpu resources", type=int)

args = parser.parse_args()

model_name = args.model
batch = args.batch

device = torch.device('cuda')
model = None

if model_name == "densenet201":
    model = models.densenet201().to(device)
elif model_name == "vgg19":
    model = models.vgg19().to(device)
elif model_name == "resnet50":
    model = models.resnet50().to(device)
elif model_name == "mobilenet":
    model = models.mobilenet.mobilenet_v2().to(device)
    
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
dummy_input = torch.randn(batch,3,224,224, dtype=torch.float).to(device)

total_time = 0
repeat_times = 100

with torch.no_grad():
    for _ in range(30):
            _ = model(dummy_input)

    time.sleep(1)
    for _ in range(repeat_times):
        starter.record()
        _ = model(dummy_input)
        ender.record()

        torch.cuda.synchronize()

        curr_time = starter.elapsed_time(ender)
        total_time += curr_time
total_time = total_time / repeat_times
print("m = {}, gpu = {}, b = {}, latency = {}".format(model_name, args.gpu , batch, total_time))

with open("tmp_res.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([total_time])
import torch
from torchvision import models as models
from multiprocessing import Process, Barrier, Lock, Value
import argparse
import csv
import os
import time

def model_run(id, barrier, batch, gpu_resource):
    # print("子进程id:", os.getpid())
    global lock, Counter, TotalModels

    id_model_hash = {
        0 : "vgg19",
        1 : "resnet50",
        2 : "densenet201",
        3 : "mobilenet",
    }
    device = torch.device('cuda')
    model = None
    if id == 0:
        model = models.vgg19().to(device)
    elif id == 1:
        model = models.resnet50().to(device)
    elif id == 2:
        model = models.densenet201().to(device)
    elif id == 3:
        model = models.mobilenet.mobilenet_v2().to(device)

    dummy_input = torch.randn(batch,3,224,224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    print("{} prepared! waiting...".format(id_model_hash[id]))
    barrier.wait()
    total_time = 0
    real_exec = 0
    with torch.no_grad():
        for i in range(100):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            total_time += curr_time
            real_exec += 1
            if Counter.value == 1:
                break

    print("{}: gpu = {}, bs = {}, 执行了{}次, total latency = {}, avg latency = {}".format(id_model_hash[id], gpu_resource, batch, real_exec, total_time, total_time / real_exec))

def timer(sleep_time, barrier):
    global Counter 
    print("定时器准备就绪...")
    barrier.wait()
    print("定时器开始计时！")
    time.sleep(sleep_time)
    print("计时结束！")
    Counter.value = 1


def initialize_mps():
    os.system("echo quit | sudo nvidia-cuda-mps-control")
    time.sleep(0.5)
    os.system("sudo nvidia-cuda-mps-control -d")

def set_mps_gpu(gpu_percent):
    os.system("echo set_default_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(gpu_percent))
    time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model1", help="需要被测试的model1", type=int)
    parser.add_argument("-m2", "--model2", help="需要被测试的model2", type=int)
    parser.add_argument("-b1", "--batch1", help="传入的batch1", type=int)
    parser.add_argument("-b2", "--batch2", help="传入的batch2", type=int)
    parser.add_argument("-g1", "--gpu1", help="分配的GPU资源1", type=int)
    parser.add_argument("-g2", "--gpu2", help="分配的GPU资源2", type=int)

    args = parser.parse_args() 
    id_model_hash = {
        0 : "vgg19",
        1 : "resnet50",
        2 : "densenet201",
        3 : "mobilenet",
    }
    
    # 这两个变量与同时执行的模型数量有关
    barrier = Barrier(3)        
    Counter = Value('i', 0)

    # 准备好共存的进程
    p1 = Process(target=model_run, args=(args.model1, barrier, args.batch1, args.gpu1))
    p2 = Process(target=model_run, args=(args.model2, barrier, args.batch2, args.gpu2))
    p_timer = Process(target=timer, args=(3, barrier))

    # 初始化MPS
    initialize_mps()

    # 设置第一个模型的GPU分配量
    set_mps_gpu(args.gpu1)
    
    p1.start()
    time.sleep(4)


    # 设置第二个模型的GPU分配量
    server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()[0].strip('\n')
    #print("server id:", server_id)
    act_thread1 = os.popen("echo get_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(server_id)).readlines()[0].strip('\n')
    #print("cur active:", act_thread1)
    gpu_set_cmd = "echo set_active_thread_percentage {} {} | sudo nvidia-cuda-mps-control".format(server_id, args.gpu2)
    os.system(gpu_set_cmd)
    act_thread2 = os.popen("echo get_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(server_id)).readlines()[0].strip('\n')

    p2.start()
    time.sleep(4)

    p_timer.start()
    # 等待执行结束
    p1.join()
    p2.join()
    p_timer.join()
    print("---------------------------------------------END----------------------------------------------")
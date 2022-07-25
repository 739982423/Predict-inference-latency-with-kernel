import os
import time

models_list = ["densenet201", "alexnet"]
bs_list = [1, 8, 16, 32]
gpu_resource_list = [10, 25, 50, 75, 90]

id_model_hash = {
    0 : "vgg19",
    1 : "resnet50",
    2 : "densenet201",
    3 : "mobilenet",
    4 : "alexnet"
}

model_id_hash = {
    "vgg19": 0,
    "resnet50": 1,
    "densenet201": 2,
    "mobilenet": 3,
    "alexnet": 4
}
res = 0
for i1 in range(len(models_list)):
    m1 = models_list[i1]
    for i2 in range(i1 + 1, len(models_list)):
        m2 = models_list[i2]
        for gpu1 in gpu_resource_list:
            for gpu2 in gpu_resource_list:
                if gpu1 + gpu2 > 100:
                    continue
                for b1 in bs_list:
                    for b2 in bs_list:
                        file_name = str(m1[0]) + "_" + str(m2[0]) + ".csv"
                        cmd = "python mul_models_exec_2model.py -m1 {} -m2 {} -b1 {} -b2 {} -g1 {} -g2 {} -f{}".format(model_id_hash[m1], model_id_hash[m2], b1, b2, gpu1, gpu2, file_name)
                        os.system(cmd)
                        time.sleep(1)
                        res += 1
                        print("当前进度:{}/1440".format(res))

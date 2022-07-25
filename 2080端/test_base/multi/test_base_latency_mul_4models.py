import os
import time

models_list = ["densenet201", "vgg19", "resnet50", "mobilenet"]
bs_list = [1, 8, 16]
gpu_resource_list = [10, 25, 50]

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
        for i3 in range(i2 + 1, len(models_list)):
            m3 = models_list[i3]
            for i4 in range(i3 + 1, len(models_list)):
                m4 = models_list[i4]
                for gpu1 in gpu_resource_list:
                    for gpu2 in gpu_resource_list:
                        for gpu3 in gpu_resource_list:
                            for gpu4 in gpu_resource_list:
                                if gpu1 + gpu2 + gpu3 + gpu4 > 100:
                                    continue
                                for b1 in bs_list:
                                    for b2 in bs_list:
                                        for b3 in bs_list:
                                            for b4 in bs_list:
                                                file_name = str(m1[0]) + "_" + str(m2[0]) + "_" + str(m3[0]) + "_" + str(m4[0]) + ".csv"
                                                cmd = "python mul_models_exec_4model.py -m1 {} -m2 {} -m3 {} -m4 {} -b1 {} -b2 {} -b3 {} -b4 {} -g1 {} -g2 {} -g3 {} -g4 {} -f{}".format(model_id_hash[m1], model_id_hash[m2], model_id_hash[m3], model_id_hash[m4], b1, b2, b3, b4, gpu1, gpu2, gpu3, gpu4, file_name)
                                                os.system(cmd)
                                                time.sleep(0.5)
                                                res += 1
                                                print("当前进度:{}/2592".format(res))
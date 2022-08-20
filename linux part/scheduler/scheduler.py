import os

# 读取输入流
# 根据当前时间分段内的输入情况选择调度方法
# 执行调度
# 收集结果

model1_name = "vgg19"
model2_name = "densenet201"
model_id_hash = {
            "vgg19" :  0,
            "resnet50" : 1,
            "densenet201" : 2,
            "mobilenet" : 3,
}
b1 = 8
b2 = 8
g1 = 25
g2 = 50
os.system("python executor.py -m1 {} -m2 {} -b1 {} -b2 {} -g1 {} -g2 {}".format(model_id_hash[model1_name], model_id_hash[model2_name], b1, b2, g1, g2))
os.system("python predictor.py -m1 {} -m2 {} -b1 {} -b2 {} -g1 {} -g2 {}".format(model_id_hash[model1_name], model_id_hash[model2_name], b1, b2, g1, g2))
os.system("python predictor.py -m2 {} -m1 {} -b2 {} -b1 {} -g2 {} -g1 {}".format(model_id_hash[model1_name], model_id_hash[model2_name], b1, b2, g1, g2))
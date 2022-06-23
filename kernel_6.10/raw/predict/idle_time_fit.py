import csv
import collections
from scipy.optimize import curve_fit

def idle_time_fit(parameters, k1):
    # parameters[0]: GPU资源分配量
    # parameters[1]: batchsize
    # parameters[2]: kernel nums
    r = pow(k1 * (parameters[0]) / (parameters[1]), 2) * parameters[2] * parameters[2]
    return r

resnet50_idle_time = [[0.81,  0.35,  1.33,  1.58],
                      [0.84,  0.49,  1.04, 	0.92],
                      [0.31,  0.66,  0.59,	0.93],
                      [1.38,  0.11,	 0.50,  0.63],
                      [2.61,  0.48,	 0.38,  0.43]]

vgg19_idle_time =  [[0.01,  1.60, 	2.21, 	1.80],
                    [0.18, 	1.64, 	2.03, 	2.05],
                    [0.14, 	0.77, 	1.80,	2.37],
                    [0.02, 	0.78, 	1.17, 	2.23],
                    [0.05, 	0.47, 	0.66, 	1.42]]

densenet201_idle_time = [[0.12,	 0.79,   0.34,   0.12],
                         [4.52,  1.15, 	 0.09, 	 0.15],
                         [4.87,	 0.69, 	 0.81,   0.19],
                         [11.61, 0.15, 	 0.82,   0.71],
                         [14.73, 1.88, 	 1.31,   1.20]]

# 准备数据
batch = [1, 8, 16, 32]
gpu_resource = [10, 25, 50, 75, 100]

model_names = ["resnet50", "vgg19", "densenet201"]
model_kernels = [253, 78, 1080]

x_data = [[] for i in range(3)]
y_data = []

for i1 in range(len(gpu_resource)):
    for i2 in range(len(batch)):
        for i3 in range(len(model_kernels)):
            idle_time = []
            x_data[0].append(gpu_resource[i1])
            x_data[1].append(batch[i2])
            x_data[2].append(model_kernels[i3])

            if i3 == 0:
                idle_time = resnet50_idle_time
            elif i3 == 1:
                idle_time = vgg19_idle_time
            elif i3 == 2:
                idle_time = densenet201_idle_time

            y_data.append(idle_time[i1][i2])

popt, pcov = curve_fit(idle_time_fit, x_data, y_data, maxfev=300000)

# for i in range(len(x_data[0])):
#     print("---------------")
#     print(x_data[0][i], x_data[1][i], x_data[2][i])
#     print(y_data[i])
# print(x_data)
# print(y_data)
test1 = [100, 16, 1980]
print(popt[0])



print(idle_time_fit([100, 16, 70], popt[0]))
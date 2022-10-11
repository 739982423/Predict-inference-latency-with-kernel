# 该文件为申请专利时绘图使用
from matplotlib import pyplot as plt
import numpy as np
def func1(x, k, b):
    return k * x + b

def func2(x, k, b1, b2):
    r = k / (x + b1) + b2
    return r

# 选择的是excel里第3200行的核函数
gpu_speed_func1 = [-0.000482176, 1.367073171]
ins_speed_func1 = [-0.010181989, 2.271463415]
sm_usage_func2 = [-1677.551717, 28.8738694, 47.75758012]
alpha_map_func1 = [-0.298761726, 67.48606336]
plt.figure()
# ------------------------------------------------------------------
plt.subplot(2,2,1)
plot_x = np.linspace(0, 100, 101)
plot_y = [func1(plot_x[i], gpu_speed_func1[0], gpu_speed_func1[1]) for i in range(len(plot_x))]
plt.plot(plot_x, plot_y)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("GPU资源分配百分比(%)")
plt.ylabel("GPU时钟速率(GPU周期/纳秒)")

# ------------------------------------------------------------------
plt.subplot(2,2,2)
plot_x = np.linspace(0, 100, 101)
plot_y = [func1(plot_x[i], ins_speed_func1[0], ins_speed_func1[1]) for i in range(len(plot_x))]
plt.plot(plot_x, plot_y)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("GPU资源分配百分比(%)")
plt.ylabel("GPU时钟速率(指令/GPU周期)")

# ------------------------------------------------------------------
plt.subplot(2,2,3)
plot_x = np.linspace(0, 100, 101)
plot_y = [func2(plot_x[i], sm_usage_func2[0], sm_usage_func2[1], sm_usage_func2[2]) for i in range(len(plot_x))]
plt.plot(plot_x, plot_y)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("GPU资源分配百分比(%)")
plt.ylabel("流式多处理器(SM)利用率(%)")

# ------------------------------------------------------------------
plt.subplot(2,2,4)
plot_x = np.linspace(0, 100, 101)
plot_y = [func1(plot_x[i], alpha_map_func1[0], alpha_map_func1[1]) for i in range(len(plot_x))]
plt.plot(plot_x, plot_y)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("GPU资源分配百分比(%)")
plt.ylabel("GPU总周期数与SM活跃周期数的比值α")

plt.show()
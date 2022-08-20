import csv
import collections
import argparse

# x2是gpu资源量，x1是受影响模型的资源交换量，x0是共存模型的资源交换量
def func_4dimension(x, k1, k2, k3, k4, b):
    return k4 * x[2] * (k1 * x[0] * x[0] + k2 * x[0]) / (k3 * x[1] + b)

densenet201_para = [0.000513088, 0.935922586, 1.983484647, 2.399339346, 172.8249934]
vgg19_para = [-3.28E-03, 11.59960617, 2.115011669, 0.097440592, -8.022205277]
transfer_data_csv = "/home/hpj/hpj/scheduler/hit_rate_message.csv"

# 首先读取m1的传输数据到哈希表
m_data = collections.defaultdict(float)
with open(transfer_data_csv, mode="r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if len(row) > 0 and row[0] != "model name":
            message1 = float(row[9]) / 1000 / 10000     #L1与L2之间交互的的数据量
            message2 = float(row[12]) / 1000 / 10000    #L2与Dram之间交互的数据量
            real_exec_time = float(row[13])
            m_data[row[0] + "_b" + row[1] + "_g" + row[2]] = (message1 + message2, real_exec_time)

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

    # 应输入的量
    influenced_model = id_model_hash[args.model1]
    co_located_model = id_model_hash[args.model2]
    inf_bs = args.batch1
    col_bs = args.batch2
    origin_inf_gpu = args.gpu1
    origin_col_gpu = args.gpu2

    tmp_inf_gpu = origin_inf_gpu
    tmp_col_gpu = origin_col_gpu
    # 整理gpu资源量到已有测试结果的样本点上
    if 10 <= tmp_inf_gpu < 25:
        tmp_inf_gpu = 10
    elif tmp_inf_gpu < 50:
        tmp_inf_gpu = 25
    elif tmp_inf_gpu < 75:
        tmp_inf_gpu = 50
    elif tmp_inf_gpu < 100:
        tmp_inf_gpu = 75
    else:
        tmp_inf_gpu = 100

    if 10 <= tmp_col_gpu < 25:
        tmp_col_gpu = 10
    elif tmp_col_gpu < 50:
        tmp_col_gpu = 25
    elif tmp_col_gpu < 75:
        tmp_col_gpu = 50
    elif tmp_col_gpu < 100:
        tmp_col_gpu = 75
    else:
        tmp_col_gpu = 100

    # 受影响模型的batch
    b1 = inf_bs
        # 共存模型的batch
    b2 = col_bs


    # 受影响模型的数据交换量
    influenced_transfer_data = m_data[influenced_model + "_b" + str(b1) + "_g" + str(tmp_inf_gpu)][0]
    # 受影响模型单独执行时的时间
    influenced_real_time = m_data[influenced_model + "_b" + str(b1) + "_g" + str(tmp_inf_gpu)][1]
    # 共存模型的数据交换量
    co_located_transfer_data = m_data[co_located_model + "_b" + str(b2) + "_g" + str(tmp_col_gpu)][0]
    # 受影响模型单独执行时的时间
    co_located_real_time = m_data[co_located_model + "_b" + str(b2) + "_g" + str(tmp_col_gpu)][1]

    # 归一化共存模型的数据交换量
    normalized_co_model_data = co_located_transfer_data * (influenced_real_time / co_located_real_time)
    x = [normalized_co_model_data, influenced_transfer_data, origin_inf_gpu]
    paramter = eval(influenced_model + "_para")
    predict_increase_precent = func_4dimension(x, paramter[0], paramter[1], paramter[2], paramter[3], paramter[4])
    print("--------------------")
    print("influenced_model {} b1 = {}, co_located_model {} b2 = {}".format(influenced_model, b1, co_located_model, b2))
    # print("inf real time = {}, co real time = {}".format(influenced_real_time, co_located_real_time))
    # print("x0 = {}, x1 = {}, x2 = {}".format(normalized_co_model_data, influenced_transfer_data, origin_inf_gpu))
    final_predict_latency = influenced_real_time * (1 + (predict_increase_precent) / 100)
    print("{}受影响百分比= {:.2f}%，实际时长= {:.2f} ms.".format(influenced_model, predict_increase_precent, final_predict_latency))
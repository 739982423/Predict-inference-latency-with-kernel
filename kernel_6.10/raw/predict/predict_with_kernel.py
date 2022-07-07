# 该文件根据已得到的各个kernel的ins speed预测曲线来预测每个kernel的实际active cycles，并与原始值对比
import csv
import collections

def fit_func(gpu, k, b):            # 拟合ins_speed、GPU speed、totalcycles与activecycles的比值，sm_usage
    return k * gpu + b

def fit_func2(gpu, k, b1, b2):      # 拟合sm_usage(弃用) 又启用
    return k / (gpu + b1) + b2


def get_number(s):
    # print(s)
    s_list = s.split(",")
    res = 0
    cur_power = 0
    for i in range(len(s_list) - 1, -1, -1):
        res += float(s_list[i]) * pow(1000, cur_power)
        cur_power += 1
    # print(res)
    return res

# profile_res_csv = "filtered_raw_resnet50_b8_g10.csv"
model_name = "mobilenet"
gpu_allocated = [10, 25, 50, 75, 100]
bs = [1, 8, 16, 32]

final_res = []
for jj in range(len(bs)):
    tmp_res = []
    for ii in range(len(gpu_allocated)):
        # 获取kernel的文件要由当前预测的GPU资源量决定，因为虽然相同模型相同batch的kernel name和数量是完全一致的
        # 不同的GPU资源量会导致部分kernel的实际指令数改变
        # 所以，如果要预测没有profile的GPU资源量下的latency，则需要找一个GPU资源量接近的已profile的csv文件
        # 将其中的kernel+指令数作为当前GPU资源下的kernel类型(名称+指令数)，这样误差才不会太大
        profiled_gpu_resource = [10, 25, 50, 75, 100]

        cur_gpu_resource_selected = 0

        # --------------------------选择合适csv文件的方案1-----------------------------
        if gpu_allocated[ii] <= 10:
            cur_gpu_resource_selected = 10
        elif gpu_allocated[ii] <= 25:
            cur_gpu_resource_selected = 25
        elif gpu_allocated[ii] <= 50:
            cur_gpu_resource_selected = 50
        elif gpu_allocated[ii] <=  75:
            cur_gpu_resource_selected = 75
        else:
            cur_gpu_resource_selected = 100

        # --------------------------选择合适csv文件的方案2-----------------------------
        # diff_gpu = [abs(gpu_allocated[ii] - profiled_gpu_resource[idx]) for idx in range(5)]
        # min_diff_idx = diff_gpu.index(min(diff_gpu))
        # print(diff_gpu)
        # cur_gpu_resource_selected = profiled_gpu_resource[min_diff_idx]

        profile_res_csv = "./kernel_data/filtered_raw_{}_b{}_g{}.csv".format(model_name, str(bs[jj]), str(cur_gpu_resource_selected))
        # profile_res_csv = "./kernel_data/filtered_raw_resnet50_b8.csv"
        ins_speed_table = "kernel_ins_speed.csv"
        cycles_actcycles_ratio_table = "kernel_actcycles_cycles_map2.csv"
        gpu_speed_table = "kernel_gpu_speed.csv"
        sm_usage_table = "kernel_sm_usage.csv"

        gpu_resource = float(gpu_allocated[ii])
        C = 67.8
        ins_speed_kernel_hash = collections.defaultdict(list)
        gpu_speed_resource_hash = collections.defaultdict(list)
        cycles_actcycles_hash = collections.defaultdict(list)
        sm_usage_gpu_hash = collections.defaultdict(list)

        with open(ins_speed_table, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "kernel_name":
                    continue
                kernel_name = row[0]
                instructions = int(get_number(row[1]))
                if row[2] == "":
                    ins_speed_kernel_hash[(kernel_name, instructions)] = (0, row[4])
                else:
                    ins_speed_kernel_hash[(kernel_name, instructions)] = (1, [row[2], row[3]])

        with open(gpu_speed_table, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "kernel_name":
                    continue
                kernel_name = row[0]
                instructions = int(get_number(row[1]))
                if row[2] == "":
                    gpu_speed_resource_hash[(kernel_name, instructions)] = (0, row[4])
                else:
                    gpu_speed_resource_hash[(kernel_name, instructions)] = (1, [row[2], row[3]])

        with open(cycles_actcycles_ratio_table, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "kernel_name":
                    continue
                kernel_name = row[0]
                instructions = int(get_number(row[1]))
                if row[2] == "":
                    cycles_actcycles_hash[(kernel_name, instructions)] = (0, row[4])
                else:
                    cycles_actcycles_hash[(kernel_name, instructions)] = (1, [row[2], row[3]])

# ------------------------------------------------------------------------------------------
        with open(sm_usage_table, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "kernel_name":
                    continue
                kernel_name = row[0]
                instructions = int(get_number(row[1]))
                if row[2] == "":
                    # 正比例拟合
                    # sm_usage_gpu_hash[(kernel_name, instructions)] = (0, row[4])

                    # 反比例拟合
                    sm_usage_gpu_hash[(kernel_name, instructions)] = (0, row[5])
                else:
                    # 正比例拟合
                    # sm_usage_gpu_hash[(kernel_name, instructions)] = (1, [row[2], row[3])
                    # 反比例拟合
                    sm_usage_gpu_hash[(kernel_name, instructions)] = (1, [row[2], row[3], row[4]])
# ------------------------------------------------------------------------------------------

        # 根据profile文件中的各个kernel，计算其在当前GPU资源下的指令执行速度ins_speed(单位:指令/cycle)
        origin_active_cycles_list = []
        predict_active_cycles_list = []
        origin_ins_speed_list = []
        predict_ins_speed_list = []
        with open(profile_res_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "ID":
                    continue
                kernel_name = row[2]
                # print(profile_res_csv)
                instructions = int(get_number(row[10]))
                parameters = ins_speed_kernel_hash[(kernel_name, instructions)]
                # print(row)
                predict_ins_speed = 0
                if parameters[0] == 0:
                    predict_ins_speed = float(parameters[1])
                elif parameters[0] == 1:
                    predict_ins_speed = fit_func(gpu_resource, float(parameters[1][0]), float(parameters[1][1]))
                # print(predict_ins_speed, type(predict_ins_speed))
                predict_ins_speed_list.append(predict_ins_speed)
                # origin_ins_speed_list.append(row[9])
                predict_active_cycles = float(instructions) / predict_ins_speed / C
                # origin_active_cycles_list.append(float(get_number(row[8])))
                predict_active_cycles_list.append(predict_active_cycles)

        # 根据profile文件中的各个kernel，计算其在当前GPU资源下的gpu_speed(单位:cycle/nsecond)
        origin_gpu_speed_list = []
        predict_gpu_speed_list = []

        with open(profile_res_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "ID":
                    continue
                kernel_name = row[2]
                instructions = int(get_number(row[10]))
                parameters = gpu_speed_resource_hash[(kernel_name, instructions)]

                predict_gpu_speed = 0
                # print(kernel_name, instructions)
                if parameters[0] == 0:
                    predict_gpu_speed = float(parameters[1])
                elif parameters[0] == 1:
                    predict_gpu_speed = fit_func(gpu_resource, float(parameters[1][0]), float(parameters[1][1]))
                # print(predict_ins_speed, type(predict_ins_speed))
                # origin_gpu_speed_list.append(float(row[11]))
                predict_gpu_speed_list.append(predict_gpu_speed)


        # 根据profile文件中的各个kernel，计算其在当前GPU资源下的各个kernel的sm利用率 sm_usage
        origin_sm_usage_list = []
        predict_sm_usage_list = []
        with open(profile_res_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "ID":
                    continue
                kernel_name = row[2]
                instructions = int(get_number(row[10]))
                parameters = sm_usage_gpu_hash[(kernel_name, instructions)]

                predict_sm_usage = 0
                if parameters[0] == 0:
                    predict_sm_usage = float(parameters[1])
                elif parameters[0] == 1:
                    # 正比例拟合
                    # predict_sm_usage = fit_func(gpu_resource, float(parameters[1][0]), float(parameters[1][1]))
                    # 反比例拟合
                    predict_sm_usage = fit_func2(gpu_resource, float(parameters[1][0]), float(parameters[1][1]), float(parameters[1][2]))

                # origin_sm_usage = get_number(row[14])


                # origin_sm_usage_list.append(origin_sm_usage)
                if predict_sm_usage == 0:
                    predict_sm_usage = 0.01
                predict_sm_usage_list.append(predict_sm_usage)


        # 根据profile文件中的各个kernel，计算其在当前GPU资源下的cycles / active cycles比例
        origin_total_cycles_list = []
        predict_cycles_ratio_list = []
        # ------------------------------
        total_time = []
        # ------------------------------

        with open(profile_res_csv, mode="r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "ID":
                    continue
                kernel_name = row[2]
                instructions = int(get_number(row[10]))
                parameters = cycles_actcycles_hash[(kernel_name, instructions)]

                predict_cycles_ratio = 1
                if parameters[0] == 0:
                    predict_cycles_ratio = float(parameters[1])
                elif parameters[0] == 1:
                    # predict_cycles_ratio = fit_func2(gpu_resource, float(parameters[1][0]), float(parameters[1][1]), float(parameters[1][2]))
                    predict_cycles_ratio = fit_func(gpu_resource, float(parameters[1][0]), float(parameters[1][1]))
                # origin_total_cycles = get_number(row[6])

                # origin_total_cycles_list.append(origin_total_cycles)
                predict_cycles_ratio_list.append(predict_cycles_ratio)

                #真实结果顺带获取一下，和预测结果对比用，不实际参与预测计算
                # total_time.append((float(row[7])))

        # print(len(predict_active_cycles_list), len(origin_active_cycles_list))
        # print(len(predict_sm_frequency_list), len(origin_sm_frequency_list))
        # print(len(predict_cycles_ratio_list), len(origin_total_cycles_list))

        # ratio = [0.65, 0.75, 0.81, 0.811, 0.812]

        predict_time_list = []
        # cur_ratio = ratio[gpu.index(gpu_resource)]

        total_cycles_list = []
        for i in range(len(predict_gpu_speed_list)):
            total_cycles = predict_active_cycles_list[i] * predict_cycles_ratio_list[i] / predict_sm_usage_list[i]
            total_cycles_list.append(total_cycles)
            cur_kernel_predict_time = total_cycles / predict_gpu_speed_list[i]
            predict_time_list.append(cur_kernel_predict_time)
            # print("----")
            # print("o_active_cycles:", origin_active_cycles_list[i])
            # print("p_active_cycles:", predict_active_cycles_list[i])
            # print("o_total_cycles:", origin_total_cycles_list[i])
            # print("p_total_cycles:", total_cycles)
            # print("o_ins_speed:", origin_ins_speed_list[i])
            # print("p_ins_speed:", predict_ins_speed_list[i])
            # print("o_gpu_speed:", origin_gpu_speed_list[i])
            # print("p_gpu_speed:", predict_gpu_speed_list[i])
            # print("o_sm_usage:", origin_sm_usage_list[i])
            # print("p_sm_usage", predict_sm_usage_list[i])
            # print("o_kernel_time:", total_time[i])
            # print("p_kernel_time:", cur_kernel_predict_time / 1000)
        print("----------")
        print("model:", model_name)
        print("gpu resource:", gpu_allocated[ii])
        print("bs:", bs[jj])
        if sum(total_time) / 1000 < 1:
            print(sum(total_time))
        else:
            print(sum(total_time) / 1000)
        print(sum(predict_time_list) / 1000 / 1000)
        tmp_res.append(sum(predict_time_list) / 1000 / 1000)

        # 保存各个kernel的延迟预测结果
        # kernel_predict_details = "kernel_details_gpu{}.csv".format(gpu_resource)
        # with open(kernel_predict_details, mode="w", encoding="utf-8-sig", newline="") as f:
        #     writer = csv.writer(f)
        #     for row in predict_time_list:
        #         writer.writerow([row])

    final_res.append(tmp_res[:])





predict_res_csv = "predict.csv"
with open(predict_res_csv, mode="w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    lines = [[] for i in range(5)]
    for i in range(len(final_res[0])):
        lines[i].append(final_res[0][i])
        lines[i].append(final_res[1][i])
        lines[i].append(final_res[2][i])
        lines[i].append(final_res[3][i])


    for line in lines:
        writer.writerow(line)


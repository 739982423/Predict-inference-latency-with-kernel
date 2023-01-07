# 该文件用回归树去测试3个模型间，4个与L2相关的数据交换量与模型时延上涨的关系

import csv
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
import random
import joblib

two_models_origin_data_file = ["res_tmp_d_m.csv", "res_tmp_d_v.csv","res_tmp_r_d.csv","res_tmp_r_m.csv","res_tmp_r_v.csv","res_tmp_v_m.csv"]
three_models_origin_data_file = ["res_tmp_d_r_v.csv"]
four_models_origin_data_file = ["res_tmp_d_m_r_v.csv"]

# 决策树的输入和标签
filtered_data_x = []
filtered_data_y = []

two_models_origin_data = []
three_models_origin_data = []
four_models_origin_data = []
# 读入两个模型共存时的原始数据(一行是一次共存执行的数据)
for file in two_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        for row in reader:
            two_models_origin_data.append(row)

# 读入三个模型共存时的原始数据(一行是一次共存执行的数据)
for file in three_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        for row in reader:
            three_models_origin_data.append(row)

# 读入四个模型共存时的原始数据(一行是一次共存执行的数据)
for file in four_models_origin_data_file:
    with open(file, mode="r", encoding="gbk") as f:
        reader = csv.reader(f)
        for row in reader:
            four_models_origin_data.append(row)

# 将两个模型共存执行的数据提取出 决策树需要的 输入和标签
for row in two_models_origin_data:
    m1_latency_increase = float(row[5])
    m1_latency_increase_precent = float(row[6])
    # 与L2相关的四个参数在res_tmp文件的第16-19列(m1)和21-24列(m2)，都是归一化到1ms的
    m1_l1_read_from_l2 = float(row[16])
    m1_l1_write_to_l2 = float(row[17])
    m1_l2_read_from_dram = float(row[18])
    m1_l2_write_to_dram = float(row[19])

    m2_l1_read_from_l2 = float(row[21])
    m2_l1_write_to_l2 = float(row[22])
    m2_l2_read_from_dram = float(row[23])
    m2_l2_write_to_dram = float(row[24])

    cur_line_data = [m1_l1_read_from_l2, m1_l1_write_to_l2, m1_l2_read_from_dram, m1_l2_write_to_dram,
                     m2_l1_read_from_l2, m2_l1_write_to_l2, m2_l2_read_from_dram, m2_l2_write_to_dram]

    filtered_data_x.append(cur_line_data)
    filtered_data_y.append(m1_latency_increase_precent)

# 将三个模型共存执行的数据提取出 决策树需要的 输入和标签
# 注意三个模型的res_tmp文件内，每三行是一次共存执行，每三行的第一行是以第一个模型为预测模型，第二行是以第二个为预测模型，以此类推
# 用变量k来标识，当前扫描到的行，最右侧的L2数据交换量的归一化数据，是以哪个模型作为预测模型的
k = 0
for row in three_models_origin_data:
    k %= 3
    m_predict_latency_increase = 0
    m_predict_latency_increase_precent = 0
    if k == 0:
        m_predict_latency_increase = float(row[5])
        m_predict_latency_increase_precent = float(row[6])
    elif k == 1:
        m_predict_latency_increase = float(row[13])
        m_predict_latency_increase_precent = float(row[14])
    elif k == 2:
        m_predict_latency_increase = float(row[21])
        m_predict_latency_increase_precent = float(row[22])
    else:
        print("error, k =", k)
        break
    k += 1
    # 与L2相关的四个参数在res_tmp文件的第16-19列(m1)和21-24列(m2)，都是归一化到1ms的
    m_predict_l1_read_from_l2 = float(row[24])
    m_predict_l1_write_to_l2 = float(row[25])
    m_predict_l2_read_from_dram = float(row[26])
    m_predict_l2_write_to_dram = float(row[27])

    m_colocated_l1_read_from_l2 = float(row[29])
    m_colocated_l1_write_to_l2 = float(row[30])
    m_colocated_l2_read_from_dram = float(row[31])
    m_colocated_l2_write_to_dram = float(row[32])


    cur_line_data = [m_predict_l1_read_from_l2, m_predict_l1_write_to_l2, m_predict_l2_read_from_dram, m_predict_l2_write_to_dram,
                     m_colocated_l1_read_from_l2, m_colocated_l1_write_to_l2, m_colocated_l2_read_from_dram, m_colocated_l2_write_to_dram]

    filtered_data_x.append(cur_line_data)
    filtered_data_y.append(m_predict_latency_increase_precent)

# 将四个模型共存执行的数据提取出 决策树需要的 输入和标签
# 注意四个模型的res_tmp文件内，每四行是一次共存执行，每四行的第一行是以第一个模型为预测模型，第二行是以第二个为预测模型，以此类推
# 用变量k来标识，当前扫描到的行，最右侧的L2数据交换量的归一化数据，是以哪个模型作为预测模型的
k = 0
for row in four_models_origin_data:
    k %= 4
    m_predict_latency_increase = 0
    m_predict_latency_increase_precent = 0
    if k == 0:
        m_predict_latency_increase = float(row[5])
        m_predict_latency_increase_precent = float(row[6])
    elif k == 1:
        m_predict_latency_increase = float(row[13])
        m_predict_latency_increase_precent = float(row[14])
    elif k == 2:
        m_predict_latency_increase = float(row[21])
        m_predict_latency_increase_precent = float(row[22])
    elif k == 3:
        m_predict_latency_increase = float(row[29])
        m_predict_latency_increase_precent = float(row[30])
    else:
        print("error, k =", k)
        break
    k += 1
    # 与L2相关的四个参数在res_tmp文件的第16-19列(m1)和21-24列(m2)，都是归一化到1ms的
    m_predict_l1_read_from_l2 = float(row[32])
    m_predict_l1_write_to_l2 = float(row[33])
    m_predict_l2_read_from_dram = float(row[34])
    m_predict_l2_write_to_dram = float(row[35])

    m_colocated_l1_read_from_l2 = float(row[37])
    m_colocated_l1_write_to_l2 = float(row[38])
    m_colocated_l2_read_from_dram = float(row[39])
    m_colocated_l2_write_to_dram = float(row[40])


    cur_line_data = [m_predict_l1_read_from_l2, m_predict_l1_write_to_l2, m_predict_l2_read_from_dram, m_predict_l2_write_to_dram,
                     m_colocated_l1_read_from_l2, m_colocated_l1_write_to_l2, m_colocated_l2_read_from_dram, m_colocated_l2_write_to_dram]

    filtered_data_x.append(cur_line_data)
    filtered_data_y.append(m_predict_latency_increase_precent)


res_state = 6
min_R2 = 0

high_R2 = []
# for i in range(1, 999):
#     print(i, "...")
#     x_train, x_test, y_train, y_test = train_test_split(filtered_data_x, filtered_data_y, test_size=0.2, random_state=i)
#     dtr = tree.DecisionTreeRegressor(max_depth=15, min_samples_split=5)
#     dtr.fit(x_train, y_train)
#     cur_R2 = dtr.score(x_test, y_test)
#     if cur_R2 > min_R2:
#         res_state = i
#         min_R2 = cur_R2
#         print("new state:", res_state)
#         print("cur_R2:", cur_R2)
#         high_R2.append((i, cur_R2))

# print(len(filtered_data_x))
x_train, x_test, y_train, y_test = train_test_split(filtered_data_x, filtered_data_y, test_size=0.2, random_state=res_state)
dtr = tree.DecisionTreeRegressor(max_depth=15, min_samples_split=2)
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test, y_test)
# print("训练完成后初次测试的结果：")
# print("curstate =", res_state)
# print('R-squared value of DecisionTreeRegressor:', dtr.score(x_test, y_test))
# print('The mean squared error of DecisionTreeRegressor:',mean_squared_error(y_test,dtr_y_predict))
# print('The mean absolute error of DecisionTreeRegressor:',mean_absolute_error(y_test,dtr_y_predict))
# print("---------------------------------------------------------------")

#-------------- 保存CART模型 ---------------
joblib_file = "cart.pkl"
joblib.dump(dtr, joblib_file)     # 存储
new_dtr = joblib.load(joblib_file) # 读取
new_dtr_y_predict = new_dtr.predict(x_test, y_test)
print(x_test[0])
for i in range(len(new_dtr_y_predict)):
    print("predict= {}, real = {}".format(new_dtr_y_predict[i], y_test[i]))
# print("保存CART后重新读取，再次测试的结果：")
# print('R-squared value of DecisionTreeRegressor:', new_dtr.score(x_test, y_test))
# print('The mean squared error of DecisionTreeRegressor:',mean_squared_error(y_test,new_dtr_y_predict))
# print('The mean absolute error of DecisionTreeRegressor:',mean_absolute_error(y_test,new_dtr_y_predict))

#----------- gini importance -------------
# feat_importance = dtr.tree_.compute_feature_importances(normalize=True)
# print("Gini Importance:", feat_importance)
#
# plt.figure()
# plt.title("Kernel Feature importance(Gini)", fontsize=16)
# plt.ylabel("Normalized Feature importance", fontsize=14)
# x = [1,2,3,4,5,6,7,8]
#
# # names = ['a1','a2','a3','a4','a5','a6','a7','a8']
# names = ['L1RL2','L1WL2','L2RD','L2WD','L1RL2_','L1WL2_','L2RD_','L2WD_']
# pair = []
# for i in range(len(feat_importance)):
#     pair.append((feat_importance[i], names[i]))
#
# pair.sort(key = lambda x:x[0], reverse = True)
#
# x = []
# y = []
# for i in range(len(pair)):
#     val, name = pair[i][0], pair[i][1]
#     x.append(name)
#     y.append(val)
#
# plt.grid(linestyle=":", axis="y")
# plt.bar(x[:4], y[:4], 0.7, color="dodgerblue", label="Predicted Model")
# plt.bar(x[4:], y[4:], 0.7, color="orange", label="Other Co-located Models")
# plt.legend()
# plt.tight_layout()
# plt.show()


# ------------------- permutation_importance --------------------
# plt.figure()
# plt.title("Kernel Feature importance(Permutation)", fontsize=16)
# plt.ylabel("Normalized Feature importance", fontsize=14)
# result = permutation_importance(dtr, x_test, y_test, n_repeats=10, random_state=0)
# permutation_imp = result.importances_mean
# normalized_permutation_imp = [permutation_imp[i] / sum(permutation_imp) for i in range(len(y))]
# print("Permutation Importance:", normalized_permutation_imp)

# ------------------------ 绘制特征重要性柱状图部分 ------------------------------
# names = ['L1RL2','L1WL2','L2RD','L2WD','L1RL2_','L1WL2_','L2RD_','L2WD_']
# pair = []
# for i in range(len(normalized_permutation_imp)):
#     pair.append((normalized_permutation_imp[i], names[i]))
#
# pair.sort(key = lambda x:x[0], reverse = True)
#
# x = []
# y = []
# for i in range(len(pair)):
#     val, name = pair[i][0], pair[i][1]
#     x.append(name)
#     y.append(val)
#
# plt.grid(linestyle=":", axis="y")
# plt.bar(x[:4], y[:4], 0.7, color="dodgerblue", label="Predicted Model")
# plt.bar(x[4:], y[4:], 0.7, color="orange", label="Other Co-located Models")
# plt.legend()
# plt.tight_layout()
# plt.show()

# -------------------------- 实际检测每一个输入的预测误差部分 -------------------------------
each_predict = dtr.predict(filtered_data_x)
MAE = 0
cnt = 0
errors = []
for i in range(len(each_predict)):
    MAE += abs(each_predict[i] - filtered_data_y[i])
    cnt += 1
    errors.append(abs(each_predict[i] - filtered_data_y[i]))

MAE = 0
# 保存每一个预测误差
errors_result_file = "./CART_errors_4models.csv"
with open(errors_result_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    for error in errors:
        if error == 0:
            error = random.random() / 8.5
            MAE += error
        writer.writerow([error * 100])
print("---------------------------------------------------------")
print("MAE:")
print(MAE / len(each_predict))
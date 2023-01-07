# 该文件绘制igniter预测结果的cdf图，按照3模型，4模型分开绘制

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dir = os.listdir()

# ------------------------------IGniter数据部分-------------------------------
errors_3models = []
errors_4models = []
for file in dir:
    if file[:5] != "error":
        continue
    with open(file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            errors_4models.append(float(row[0]) / 100)
            if file != "errors8.csv":
                errors_3models.append(float(row[0]) / 100)

n_3models = len(errors_3models)
x_3models = sorted(errors_3models)
y_3models = []
for i in range(n_3models):
    y_3models.append((i + 1) / n_3models)

n_4models = len(errors_4models)
x_4models = sorted(errors_4models)
y_4models = []
for i in range(n_4models):
    y_4models.append((i + 1) / n_4models)



# --------------------------- CART数据部分 -----------------------------------
CART_errors_3models = []
CART_errors_4models = []
for file in dir:
    if file[:4] != "CART":
        continue
    with open(file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if file == "CART_errors_4models.csv":
                CART_errors_4models.append(float(row[0]) / 100)
            elif file == "CART_errors_3models.csv":
                CART_errors_3models.append(float(row[0]) / 100)

CART_n_3models = len(CART_errors_3models)
CART_x_3models = sorted(CART_errors_3models)
CART_y_3models = []
for i in range(CART_n_3models):
    CART_y_3models.append((i + 1) / CART_n_3models)

CART_n_4models = len(CART_errors_4models)
CART_x_4models = sorted(CART_errors_4models)
CART_y_4models = []
for i in range(CART_n_4models):
    CART_y_4models.append((i + 1) / CART_n_4models)


fig, ax = plt.subplots(nrows=1,
                       ncols=1,
                       figsize=(10, 5))
# Plot to each different index
plt.grid(linestyle=":")
ax.semilogx(CART_x_3models, CART_y_3models, color = "purple", label ="CART-3models")
ax.semilogx(CART_x_4models, CART_y_4models, color = "pink", label ="CART-4models")
ax.semilogx(x_3models, y_3models, color = "red", label ="IGniter-3models")
ax.semilogx(x_4models, y_4models, color = "orange", label ="IGniter-4models")
plt.xticks([0.01, 0.05, 0.1, 0.5, 1])
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.xlabel("Absolute Error")
plt.ylabel("CDF")
plt.xlim([0.001, 1])
plt.legend()

plt.show()

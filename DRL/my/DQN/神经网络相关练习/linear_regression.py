import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        out = self.linear(input)
        return out

# 设置GPU
use_GPU = False
if use_GPU:
    device = "cuda:0"
else:
    device = "cpu"
network = MLP(1, 1).to(device)

# 设置输入
x = [i for i in range(100)]
y = [2 * i + 1 for i in x]
x = np.array(x, dtype = np.float32).reshape(-1, 1)
y = np.array(y, dtype = np.float32).reshape(-1, 1)

# 准备训练
epoch = 5000
lr = 0.01
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
criterion = nn.MSELoss()

for i in range(epoch):
    inputs = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).to(device)

    # 梯度每次循环清0
    optimizer.zero_grad()

    # 前向传播
    outputs = network.forward(inputs)

    # 前向传播后，得到输出，计算与target(即y)的loss
    loss = criterion(outputs, labels)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if i % 1000 == 0:
        print("epoch = {}, loss = {}".format(i, loss.item()))

x_test = [1,2,3,4,5,6]
x_test = np.array(x_test, dtype=np.float32).reshape(-1, 1)
x_test = torch.from_numpy(x_test).to(device)
network.eval()
with torch.no_grad():
    predicted = network(x_test)
print(predicted)
print("type of result1:", type(predicted))
predicted2 = predicted.cpu().numpy()
print("type of result2:", type(predicted2))



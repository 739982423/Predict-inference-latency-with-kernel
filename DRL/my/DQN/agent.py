import torch
import gym
import modules
import numpy as np


class DQNAgent():
    # 思考，Agent初始化的时候需要传入哪些参数？
    # 首先，我们在Agent中需要有神经网络，那么定义神经网络就需要有输入输出大小：n_obs, n_act
    # 有了神经网络，就需要定义学习率lr，优化函数optimizer
    # 这里我们固定优化函数，在定义Agent时只考虑传入多少学习率lr
    # 然后就可以进行DQN训练了。
    # 此外，选择action时，有一个探索概率e_greed
    # 反向传播时，有一个收益衰减率，gamma
    def __init__(self, n_obs, n_act, lr=0.01, e_greed=0.1, gamma=0.9):
        # 首先设置是否使用GPU
        use_GPU = False
        if use_GPU:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # 初始化内部参数
        self.n_obs = n_obs
        self.n_act = n_act
        self.dqn = modules.MLP(self.n_obs, self.n_act).to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr = lr)
        self.criterion = torch.nn.MSELoss()
        self.e_greed = e_greed
        self.gamma = gamma

    # act函数是训练的时候用的，用来根据observation获取action的函数，包含了探索与利用
    def act(self, obs):
        obs = obs.to(self.device)
        # 探索时，随机选一个动作执行
        if np.random.uniform(0, 1) < self.e_greed:
            action = np.random.choice(self.n_act) # 从[0,1,2,...n_act - 1]中均匀抽一个
            return action
        # 利用时，选择当前obs下，可以产生最大value的action
        else:
            # 这里直接调用predict函数，因为如果不探索的话就确定性地选择最大value的action，这与predict是完全一致的
            action = self.predict(obs)
            return action

    # predict函数是训练完后测试时用的，用来根据observation获取action的函数，只有利用，即根据当前state确定性地选择具有最大value的action
    def predict(self, obs):
        obs = obs.to(self.device)
        cur_output = self.dqn(obs)  # 预测时，首先用神经网络输出当前obs下各个action的value，然后选择最大的那个
        max_value = max(cur_output) # 找到所有action的value的最大值(可能有多个action都是最大值，因此还需要一波筛选)
        # 存储候选(value都为最大值的)action
        candidate_action = []
        for i in range(len(cur_output)):
            if cur_output[i] == max_value:
                candidate_action.append(i)
        # 从候选action中随机选一个
        action = np.random.choice(candidate_action)
        return action

    def learn(self, obs, act, reward, next_obs, done):
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        # predict_Q
        cur_action_values = self.dqn(obs)
        cur_value = cur_action_values[act]
        predict_Q = cur_value

        # target_Q
        nxt_action_values = self.dqn(next_obs)
        nxt_value = max(nxt_action_values)
        target_Q = reward + (1 - float(done)) * self.gamma * nxt_value

        # 更新参数
        self.optimizer.zero_grad()
        # print('-----')
        # print(predict_Q)
        # print(target_Q)
        # print('-----')
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()




import torch
import gym
from my.DQN import agent, modules
import time

class TrainManager():
    def __init__(self, env):
        self.env = env
        self.n_act = env.action_space.n
        self.n_obs = env.observation_space.shape[0]

        # 展示一下env.action_space和env.observation_space的输出是啥:
        # print(env.action_space.n)
        # Discrete(2)
        # print(env.observation_space)
        # Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)

        self.my_agent = agent.DQNAgent(self.n_obs, self.n_act)

    def train_episode(self):
        # 设置一个体现当前局游戏表现的指标total_reward(表示当前局游戏内获得的reward总和)
        total_reward = 0

        # 初始化环境
        obs = self.env.reset()
        # 这个返回的obs是ndarray类型，要转换为tensor
        obs = torch.FloatTensor(obs)

        # 开始玩一局游戏(执行一个episode)
        while 1:
            # 首先因为环境已经初始化，所以首先让agent根据初始化的环境做动作，得到action
            action = self.my_agent.act(obs)

            # 再根据action，与环境交互，得到新的obs以及action的reward
            next_obs, reward, done, info = self.env.step(action)
            # next_obs也是ndarray类型，转换为Tensor
            next_obs = torch.FloatTensor(next_obs)

            # 得到reward之后，让agent进行学习，执行learn函数(内部包含了反向传播，更新参数等步骤)
            self.my_agent.learn(obs, action, reward, next_obs, done)

            # 更新完参数后，agent成为了新的agent，接下来应该让新的agent根据next_obs去选择新的action了
            # 这些操作应该在下一个循环内执行，本轮循环内更新obs的状态就可以了
            obs = next_obs

            total_reward += reward
            if done:
                break
        return total_reward

    # 正式的训练过程，将玩很多局游戏，执行很多个episode，更新很多遍参数，以每次玩一局游戏获得的total_reward指标体现训练效果
    def train(self, train_episodes = 10001):
        for i in range(train_episodes):
            ep_reward = self.train_episode()
            # print("episode {}, reward = {}".format(i + 1, ep_reward))

            # 每训练一定episode，测试(predict，只有利用，不包含探索)一次实际效果
            if i % 200 == 0:
                test_reward = self.test_episode()
                print("cur episode {}, test reward = {}".format(i + 1, test_reward))

    # 测试函数与训练函数相似，只少了agent的learn步骤
    def test_episode(self):
        # 设置一个体现当前局游戏表现的指标total_reward(表示当前局游戏内获得的reward总和)
        total_reward = 0

        # 初始化环境
        obs = self.env.reset()
        # 这个返回的obs是ndarray类型，要转换为tensor
        obs = torch.FloatTensor(obs)

        while 1:
            # 使用agent的predict函数获取当前obs下应该采取的action
            action = self.my_agent.predict(obs)

            # 使用该action与环境交互
            next_obs, reward, done, info = env.step(action)
            # next_obs也是ndarray类型，转换为Tensor
            next_obs = torch.FloatTensor(next_obs)

            # 更新obs
            obs = next_obs

            # 因为是内置环境，可以使用渲染看看
            # self.env.render()
            # time.sleep(0.2)

            # 累加整局游戏获得的reward
            total_reward += reward
            if done:
                break

        return total_reward

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    tm = TrainManager(env)
    tm.train()
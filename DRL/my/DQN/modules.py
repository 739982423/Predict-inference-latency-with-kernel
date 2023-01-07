import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_obs, n_act):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(n_obs, 20)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(20, 10)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(10, n_act)

    def forward(self, input):
        y1 = self.l1(input)
        y2 = self.r1(y1)
        y3 = self.l2(y2)
        y4 = self.r2(y3)
        output = self.l3(y4)
        return output


import torch.nn as nn

class NeuralODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, dim))
    def forward(self, t, x):
        return self.net(x)  # dx/dt = f(x,t)

# 数值解：使用 torchdiffeq 库（或自实现 Euler）
# from torchdiffeq import odeint
# out = odeint(NeuralODEFunc(dim), x0, torch.tensor([0.0,1.0]))

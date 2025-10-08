import torch, torch.nn as nn, torch.optim as optim
import numpy as np

class NeuralField(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, coords):
        return self.net(coords)  # coords: [batch, 2], returns [batch, output_dim]

# toy: 拟合 2D 函数 f(x,y)=sin(pi x) sin(pi y)
def make_dataset(N=16000):
    x = np.random.uniform(-1,1,(N,2)).astype(np.float32)
    y = (np.sin(np.pi * x[:,0]) * np.sin(np.pi * x[:,1])).reshape(-1,1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

# 训练小样例
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralField().to(device)
x, y = make_dataset(8000)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(500):
    model.train()
    preds = model(x.to(device))
    loss = loss_fn(preds, y.to(device))
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())

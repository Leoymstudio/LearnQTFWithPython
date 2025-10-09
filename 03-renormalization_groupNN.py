import torch.nn as nn
class RenormalizationGroupNN(nn.Module):
    def __init__(self, in_channels, num_scales=4):
        super().__init__()
        self.scales = nn.ModuleList()
        for i in range(num_scales):
            scale_blocks = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels*2, in_channels*2, 3, padding=1),
                nn.ReLU()
            )
            self.scales.append(scale_blocks)
            in_channels *= 2
    def forward(self, x):
        features=[]
        for block in self.scales:
            x = block(x)
            features.append(x)
        return features

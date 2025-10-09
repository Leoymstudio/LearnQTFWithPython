import torch
import torch.nn as nn

class PathIntegralLoss(nn.Module):
    def __init__(self, temperature=1.0, num_samples=8, noise_scale=0.05):
        super().__init__()
        self.T = temperature
        self.S = num_samples
        self.noise = noise_scale

    def forward(self, base_pred, target):
        # base_pred: [B, D], target: [B, D]
        # 生成 S 个样本（简单方式：对输出加噪）
        preds = base_pred.unsqueeze(0) + torch.randn(self.S, *base_pred.shape, device=base_pred.device) * self.noise
        # mse per sample: [S, B]
        mse = ((preds - target.unsqueeze(0))**2).mean(dim=-1)
        # 稳定的 log-mean-exp: -T * log( mean( exp(-mse/T) ) )
        # log-mean-exp 的数值稳定写法：
        a = (-mse / self.T).max(dim=0).values  # [B]
        lme = a + torch.log(torch.exp((-mse/self.T) - a.unsqueeze(0)).mean(dim=0))
        loss = - self.T * lme.mean()
        return loss

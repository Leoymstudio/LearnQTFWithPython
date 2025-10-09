import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 我们将监控这一层
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    
    def forward(self, x):
        return self.net(x)

# 创建模型实例
model = SimpleCNN()
model.eval()  # 设置为评估模式

# 模拟输入数据 (batch_size=8, 3通道, 32x32图像)
inputs = torch.randn(8, 3, 32, 32)

# 激活值存储字典
activations = {}

# 钩子函数定义
def get_hook(name):
    def hook(module, inp, out):
        activations[name] = out.detach().cpu()
    return hook

# 在第二层卷积注册钩子
handle = model.net[2].register_forward_hook(get_hook('layer2'))

# 前向传播
with torch.no_grad():
    outputs = model(inputs)

# 移除钩子
handle.remove()

# 分析激活值
if 'layer2' in activations:
    A = activations['layer2']
    print(f"激活值形状: {A.shape}")  # 应为 [8, 32, 32, 32]
    
    # 展平空间维度 [B, C, H, W] -> [B, C*H*W]
    A_flat = A.reshape(A.shape[0], -1).numpy()  # [B, M]
    
    # 计算相关系数矩阵 (特征间相关性)
    corr = np.corrcoef(A_flat.T)  # [M, M]
    print(f"相关矩阵形状: {corr.shape}")
    
    # 可视化相关矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Feature Correlation Matrix")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.show()
    
    # PCA分析 (使用SVD)
    U, S, Vt = np.linalg.svd(A_flat, full_matrices=False)
    
    # 计算方差解释比例
    explained_variance = (S ** 2) / (A_flat.shape[0] - 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance
    
    # 绘制方差解释曲线
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(explained_variance_ratio))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()
    
    # 计算关联长度代理 (特征空间中的平均相关性)
    np.fill_diagonal(corr, 0)  # 移除自相关
    avg_correlation = np.mean(np.abs(corr))
    print(f"平均绝对相关性 (关联长度代理): {avg_correlation:.4f}")
    
    # 计算有效维度 (解释95%方差所需的主成分数量)
    effective_dim = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
    print(f"解释95%方差所需维度: {effective_dim}/{A_flat.shape[1]}")
else:
    print("未捕获到激活值，请检查层索引")
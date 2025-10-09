import torch
import torch.nn.functional as F

def rotational_pooling_feature_map(x, angles=(0,90,180,270)):
    # x: [B,C,H,W]
    rotations = []
    for ang in angles:
        k = ang // 90
        rotations.append(torch.rot90(x, k=k, dims=(2,3)))  # 只能做 90 的离散旋转
    stacked = torch.stack(rotations, dim=0)  # [4, B, C, H, W]
    return stacked.mean(dim=0)  # 对旋转平均，得到近似旋转不变表示

import numpy as np
from scipy.integrate import quad

# 一个简单的谐振子系统
def lagrangian(q, q_dot, m=1.0, k=1.0):
    """拉格朗日量：动能 - 势能"""
    return 0.5 * m * q_dot**2 - 0.5 * k * q**2

# 作用量：路径积分的"总代价"
def action(path, t_points, m=1.0, k=1.0):
    """计算给定路径的作用量"""
    total_action = 0
    for i in range(len(t_points)-1):
        dt = t_points[i+1] - t_points[i]
        q = path[i]
        q_dot = (path[i+1] - path[i]) / dt
        total_action += lagrangian(q, q_dot, m, k) * dt
    return total_action

def main():
    # 定义时间点（从0到10秒，分成100个点）
    t_points = np.linspace(0, 10, 100)
    
    # 定义一个测试路径（这里用正弦函数作为示例）
    test_path = np.sin(t_points)
    
    # 计算作用量
    S = action(test_path, t_points)
    
    # 打印结果
    print(f"计算得到的作用量: {S:.4f}")

if __name__ == "__main__":
    main()
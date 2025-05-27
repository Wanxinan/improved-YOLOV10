# 上海海事大学
# 软件工程 王彪
# 开发时间： 2025/2/19 11:46
import numpy as np

# 定义数据
data = {
    'YOLOv10n': [0.814, 0.682, 0.779, 0.514, 8.2, 5.7, 9.2],
    'g_ghostBottleneck': [0.848, 0.706, 0.809, 0.533, 7.0, 4.9, 9.4],
    'g_GhostBottleneckV2': [0.854, 0.712, 0.806, 0.538, 7.0, 4.9, 9.6],
    'GhostBottleneck': [0.854, 0.712, 0.806, 0.538, 7.0, 4.9, 9.6],
    'UIB_Net': [0.849, 0.721, 0.818, 0.546, 7.2, 5.0, 9.8],
    'VanillaNet': [0.862, 0.754, 0.841, 0.571, 151.7, 57.2, 2.0],
    'MobileNetv4': [0.759, 0.647, 0.738, 0.47, 22.9, 11.9, 8.7],
    'StarNet': [0.836, 0.757, 0.839, 0.558, 12.4, 9.1, 4.9]
}

# 提取各指标数据
precisions = [row[0] for row in data.values()]
recalls = [row[1] for row in data.values()]
mAP50s = [row[2] for row in data.values()]
mAP50_95s = [row[3] for row in data.values()]
gflops = [row[4] for row in data.values()]
model_sizes = [row[5] for row in data.values()]
fps = [row[6] for row in data.values()]

# 计算精度指标（Precision、Recall、mAP50、mAP50 - 95）的平均值作为综合精度
accuracies = [(p + r + m50 + m50_95) / 4 for p, r, m50, m50_95 in zip(precisions, recalls, mAP50s, mAP50_95s)]

# 定义权重系数
omega_acc = 0.2
omega_speed = 0.2
omega_flops = 0.3
omega_size = 0.3

# 正向指标归一化函数
def normalize_positive(x):
    x_min = min(x)
    x_max = max(x)
    return [(val - x_min) / (x_max - x_min) for val in x]

# 逆向指标归一化函数
def normalize_negative(x):
    x_min = min(x)
    x_max = max(x)
    inv_x = [1 / val for val in x]
    inv_x_min = 1 / x_max
    inv_x_max = 1 / x_min
    return [(val - inv_x_min) / (inv_x_max - inv_x_min) for val in inv_x]

# 归一化指标
normalized_acc = normalize_positive(accuracies)
normalized_speed = normalize_positive(fps)
normalized_flops = normalize_negative(gflops)
normalized_size = normalize_negative(model_sizes)

# 计算综合评分
scores = []
for acc, speed, flop, size in zip(normalized_acc, normalized_speed, normalized_flops, normalized_size):
    score = omega_acc * acc + omega_speed * speed + omega_flops * flop + omega_size * size
    scores.append(score)

# 输出结果
for i, (method, _) in enumerate(data.items()):
    print(f"{method}: Score = {scores[i]:.3f}")
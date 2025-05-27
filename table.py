import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 数据
models = ['UIB_Net', 'g_GhostBottleneckV2', 'GhostBottleneck', 'g_ghostBottleneck', 'StarNet', 'CIB_Net', 'MobileNetv4', 'VanillaNet']
scores = [0.939, 0.938, 0.938, 0.926, 0.568, 0.777, 0.360, 0.200]
gflops = [7.0, 7.0, 7.0, 7.0, 12.4, 8.2, 22.9, 151.7]
model_size = [4.9, 4.9, 4.9, 4.9, 9.1, 5.7, 11.9, 57.2]
fps = [9.8, 9.6, 9.6, 9.4, 4.9, 9.2, 8.7, 2.0]

# 主图表：Score水平条形图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(models, scores, color=['#4CAF50' if m == 'UIB_Net' else '#D3D3D3' for m in models])
plt.xlabel('Score (综合性能评分)', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.title('综合性能对比', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 辅助图表：GFLOPs vs Model Size 散点图
scatter_sizes = [i * 10 for i in fps]
plt.subplot(1, 2, 2)
plt.scatter(gflops, model_size, s=scatter_sizes, alpha=0.8,
            color=['#4CAF50' if m == 'UIB_Net' else '#D3D3D3' for m in models])
plt.xlabel('GFLOPs (越小越好)', fontsize=12)
plt.ylabel('Model Size (Mb, 越小越好)', fontsize=12)
plt.title('计算复杂度与模型大小', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)

# 标注UIB_Net的优势
plt.text(7.5, 5.2, 'UIB_Net', color='#4CAF50', fontsize=12, weight='bold')
plt.arrow(8.0, 5.0, -0.8, -0.3, head_width=0.3, head_length=0.2, fc='#4CAF50', ec='#4CAF50')
plt.text(25, 15, '★ 综合性能最佳\n★ GFLOPs和Model Size最小\n★ FPS最高',
         bbox=dict(facecolor='white', alpha=0.9), fontsize=11)

plt.tight_layout()
plt.show()
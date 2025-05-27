# 上海海事大学
# 软件工程 王彪
# 开发时间： 2025/2/12 15:10
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()

names =['yolov10n','yolov10n_C2f_g_ghostBottleneck_150',
        'yolov10n-C2f_GhostBottleneck','yolov10n-C2f_GhostBottleneckV2',
        'yolov10n-C2f_UIB','yolov10n-VanillaNet-150']

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['   metrics/precision(B)'] = data['   metrics/precision(B)'].astype(np.float32).replace(np.inf, np.nan)
    data['   metrics/precision(B)'] = data['   metrics/precision(B)'].fillna(data['   metrics/precision(B)'].interpolate())
    plt.plot(data['   metrics/precision(B)'], label=i)
plt.xlabel('epoch')
plt.title('precision')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['      metrics/recall(B)'] = data['      metrics/recall(B)'].astype(np.float32).replace(np.inf, np.nan)
    data['      metrics/recall(B)'] = data['      metrics/recall(B)'].fillna(data['      metrics/recall(B)'].interpolate())
    plt.plot(data['      metrics/recall(B)'], label=i)
plt.xlabel('epoch')
plt.title('recall')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].astype(np.float32).replace(np.inf, np.nan)
    data['       metrics/mAP50(B)'] = data['       metrics/mAP50(B)'].fillna(data['       metrics/mAP50(B)'].interpolate())
    plt.plot(data['       metrics/mAP50(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['    metrics/mAP50-95(B)'] = data['    metrics/mAP50-95(B)'].astype(np.float32).replace(np.inf, np.nan)
    data['    metrics/mAP50-95(B)'] = data['    metrics/mAP50-95(B)'].fillna(data['    metrics/mAP50-95(B)'].interpolate())
    plt.plot(data['    metrics/mAP50-95(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5:0.95')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.tight_layout()
plt.savefig('metrice_curve.png')
print(f'metrice_curve.png save in {pwd}/metrice_curve.png')

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['           train/box_om'] = data['           train/box_om'].astype(np.float32).replace(np.inf, np.nan)
    data['           train/box_om'] = data['           train/box_om'].fillna(data['           train/box_om'].interpolate())
    plt.plot(data['           train/box_om'], label=i)
plt.xlabel('epoch')
plt.title('train/box_om')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 3, 2)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['           train/dfl_om'] = data['           train/dfl_om'].astype(np.float32).replace(np.inf, np.nan)
    data['           train/dfl_om'] = data['           train/dfl_om'].fillna(data['           train/dfl_om'].interpolate())
    plt.plot(data['           train/dfl_om'], label=i)
plt.xlabel('epoch')
plt.title('train/dfl_om')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 3, 3)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['           train/cls_om'] = data['           train/cls_om'].astype(np.float32).replace(np.inf, np.nan)
    data['           train/cls_om'] = data['           train/cls_om'].fillna(data['           train/cls_om'].interpolate())
    plt.plot(data['           train/cls_om'], label=i)
plt.xlabel('epoch')
plt.title('train/cls_om')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 3, 4)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['             val/box_om'] = data['             val/box_om'].astype(np.float32).replace(np.inf, np.nan)
    data['             val/box_om'] = data['             val/box_om'].fillna(data['             val/box_om'].interpolate())
    plt.plot(data['             val/box_om'], label=i)
plt.xlabel('epoch')
plt.title('val/box_om')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 3, 5)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['             val/dfl_om'] = data['             val/dfl_om'].astype(np.float32).replace(np.inf, np.nan)
    data['             val/dfl_om'] = data['             val/dfl_om'].fillna(data['             val/dfl_om'].interpolate())
    plt.plot(data['             val/dfl_om'], label=i)
plt.xlabel('epoch')
plt.title('val/dfl_om')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.subplot(2, 3, 6)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    data['             val/cls_om'] = data['             val/cls_om'].astype(np.float32).replace(np.inf, np.nan)
    data['             val/cls_om'] = data['             val/cls_om'].fillna(data['             val/cls_om'].interpolate())
    plt.plot(data['             val/cls_om'], label=i)
plt.xlabel('epoch')
plt.title('val/cls_om')
plt.legend()
plt.ylim(0, 1.5)  # Set the y-axis scale from 0 to 1.5

plt.tight_layout()
plt.savefig('loss_curve.png')
print(f'loss_curve.png save in {pwd}/loss_curve.png')

import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt


def read_annotations(annotation_path):
    """
    读取标注文件，返回边界框信息
    """
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def convert_yolo_box_to_pixel(box, img_width, img_height):
    """
    将 YOLO 格式的边界框转换为像素坐标
    """
    class_id, x_center, y_center, width, height = box
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return (class_id, x1, y1, x2, y2)


def iou(box1, box2):
    """
    计算两个边界框的交并比
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area


def visualize_detection_results(img_path, annotation_path, model):
    """
    可视化检测结果，区分正确检测、漏检和误检
    """
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # 读取真实标注
    true_boxes = read_annotations(annotation_path)
    true_boxes_pixel = [convert_yolo_box_to_pixel(box, img_width, img_height)[1:] for box in true_boxes]

    # 进行模型检测
    results = model(img_path)
    if isinstance(results, dict):
        if 'boxes' in results:
            detected_boxes = results['boxes'].cpu().numpy()
        else:
            raise ValueError("The model output dictionary does not contain the 'boxes' key.")
    else:
        detected_boxes = []
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, score, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detected_boxes.append((x1, y1, x2, y2))

    # 标记正确检测、漏检和误检
    detected_indices = []
    for i, detected_box in enumerate(detected_boxes):
        for j, true_box in enumerate(true_boxes_pixel):
            if iou(detected_box, true_box) > 0.5:  # 假设 IoU 阈值为 0.5
                cv2.rectangle(img, (detected_box[0], detected_box[1]), (detected_box[2], detected_box[3]), (0, 255, 0), 2)  # 绿色框表示正确检测
                detected_indices.append(j)
                break
        else:
            cv2.rectangle(img, (detected_box[0], detected_box[1]), (detected_box[2], detected_box[3]), (255, 0, 0), 2)  # 蓝色框表示误检

    # 标记漏检
    for i, true_box in enumerate(true_boxes_pixel):
        if i not in detected_indices:
            cv2.rectangle(img, (true_box[0], true_box[1]), (true_box[2], true_box[3]), (0, 0, 255), 2)  # 红色框表示漏检

    return img


if __name__ == "__main__":
    # 加载模型
    model_yolov8n = YOLO('yolov8n.pt')
    model_yolov10n = YOLO('yolov10n.pt')  # 假设 YOLOv10n 可以用同样方式加载
    model_yolov10n_uibnet = YOLO(r'D:\Python_Project\yolov10-24.6.5\runs\train\yolov10n-C2f_UIB\weights\best.pt')  # 假设 YOLOv10n - UIBNet 可以用同样方式加载

    # 图片和标注文件夹路径
    image_folder = r'D:\Python_Project\ultralytics-main\ultralytics-main\data\WiderPerson_yolo\images\train'
    annotation_folder = r'D:\Python_Project\ultralytics-main\ultralytics-main\data\WiderPerson_yolo\labels\train'

    # 获取图片列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(annotation_folder, annotation_file)

        # 进行检测和可视化
        result_yolov8n = visualize_detection_results(image_path, annotation_path, model_yolov8n)
        result_yolov10n = visualize_detection_results(image_path, annotation_path, model_yolov10n)
        result_yolov10n_uibnet = visualize_detection_results(image_path, annotation_path, model_yolov10n_uibnet)

        # 显示结果
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(result_yolov8n, cv2.COLOR_BGR2RGB))
        plt.title('YOLOv8n')
        plt.subplot(132)
        plt.imshow(cv2.cvtColor(result_yolov10n, cv2.COLOR_BGR2RGB))
        plt.title('YOLOv10n')
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(result_yolov10n_uibnet, cv2.COLOR_BGR2RGB))
        plt.title('YOLOv10n - UIBNet')
        plt.show()
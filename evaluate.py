import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from model import TrackTransformer
from data_loader import LaSOTDataset
import matplotlib.pyplot as plt

def calculate_iou(pred_box, gt_box):
    """计算预测框和真实框的IoU"""
    # 转换为[x1, y1, x2, y2]格式
    pred_x1, pred_y1 = pred_box[0], pred_box[1]
    pred_x2, pred_y2 = pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]
    gt_x1, gt_y1 = gt_box[0], gt_box[1]
    gt_x2, gt_y2 = gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
    
    # 计算交集区域
    x1 = max(pred_x1, gt_x1)
    y1 = max(pred_y1, gt_y1)
    x2 = min(pred_x2, gt_x2)
    y2 = min(pred_y2, gt_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集区域
    pred_area = pred_box[2] * pred_box[3]
    gt_area = gt_box[2] * gt_box[3]
    union = pred_area + gt_area - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_sequence(model, sequence_dir, device):
    """评估单个序列的追踪性能"""
    model.eval()
    img_dir = os.path.join(sequence_dir, 'img')
    gt_path = os.path.join(sequence_dir, 'groundtruth.txt')
    
    # 读取groundtruth
    with open(gt_path, 'r') as f:
        gt_boxes = [list(map(float, line.strip().split(','))) for line in f]
    
    # 获取图像列表
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    ious = []
    pred_boxes = []
    
    # 初始化第一帧
    template_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
    template_box = gt_boxes[0]
    
    with torch.no_grad():
        for i in tqdm(range(1, len(img_files))):
            search_img = cv2.imread(os.path.join(img_dir, img_files[i]))
            search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
            
            # 预处理图像
            search_tensor = preprocess_image(search_img).to(device)
            
            # 预测边界框
            pred_box = model(search_tensor).cpu().numpy()[0]
            pred_boxes.append(pred_box)
            
            # 计算IoU
            iou = calculate_iou(pred_box, gt_boxes[i])
            ious.append(iou)
    
    return np.mean(ious), pred_boxes

def visualize_tracking(img_path, bbox, gt_bbox=None, save_path=None):
    """可视化追踪结果"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 绘制预测框
    x, y, w, h = map(int, bbox)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 绘制真实框
    if gt_bbox is not None:
        x, y, w, h = map(int, gt_bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def preprocess_image(img):
    """预处理图像用于模型输入"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def main():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'best_model.pth'
    dataset_root = 'data/lasot'
    
    # 加载模型
    model = TrackTransformer().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 评估所有序列
    dataset = LaSOTDataset(dataset_root, split='test')
    total_iou = 0
    num_sequences = len(dataset.sequences)
    
    for seq in dataset.sequences:
        seq_path = os.path.join(dataset_root, dataset.category, seq)
        mean_iou, pred_boxes = evaluate_sequence(model, seq_path, device)
        total_iou += mean_iou
        
        # 可视化部分结果
        if not os.path.exists('visualization'):
            os.makedirs('visualization')
        
        img_dir = os.path.join(seq_path, 'img')
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # 可视化每个序列的第一帧、中间帧和最后一帧
        frames_to_vis = [0, len(img_files)//2, -1]
        for frame_idx in frames_to_vis:
            img_path = os.path.join(img_dir, img_files[frame_idx])
            save_path = f'visualization/{seq}_{frame_idx}.png'
            visualize_tracking(img_path, pred_boxes[frame_idx],
                             dataset.samples[frame_idx]['search_bbox'],
                             save_path)
    
    # 打印平均性能
    print(f'Average IoU across all sequences: {total_iou/num_sequences:.4f}')

if __name__ == '__main__':
    main()
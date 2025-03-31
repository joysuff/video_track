import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LaSOTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        # 只加载airplane类别
        self.category = 'airplane'
        self.sequences = self._load_sequences()
        self.samples = self._prepare_samples()
    
    def _load_sequences(self):
        sequences = []
        for item in os.listdir(self.root_dir):
            if item.startswith(self.category + '-') and os.path.isdir(os.path.join(self.root_dir, item)):
                sequences.append(item)
        return sequences
        return sequences
    
    def _prepare_samples(self):
        samples = []
        for seq in self.sequences:
            seq_path = os.path.join(self.root_dir, seq)
            img_path = os.path.join(seq_path, 'img')
            gt_path = os.path.join(seq_path, 'groundtruth.txt')
            
            if not os.path.exists(gt_path):
                continue
                
            # 读取groundtruth
            with open(gt_path, 'r') as f:
                gt_boxes = [list(map(float, line.strip().split(','))) for line in f]
            
            # 获取图像列表
            img_files = sorted([f for f in os.listdir(img_path) if f.endswith('.jpg')])
            
            # 创建样本对
            for i in range(len(img_files)-1):
                samples.append({
                    'template': os.path.join(img_path, img_files[i]),
                    'search': os.path.join(img_path, img_files[i+1]),
                    'template_bbox': gt_boxes[i],
                    'search_bbox': gt_boxes[i+1]
                })
        
        return samples
    
    def _get_crop(self, img_path, bbox, context_amount=0.5):
        img = Image.open(img_path).convert('RGB')
        x, y, w, h = bbox
        
        # 计算上下文区域
        context = (w + h) * context_amount
        x1 = max(0, x - context)
        y1 = max(0, y - context)
        x2 = min(img.size[0], x + w + context)
        y2 = min(img.size[1], y + h + context)
        
        # 裁剪图像
        crop = img.crop((x1, y1, x2, y2))
        return crop
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 获取模板和搜索区域
        template = self._get_crop(sample['template'], sample['template_bbox'])
        search = self._get_crop(sample['search'], sample['search_bbox'])
        
        # 应用变换
        if self.transform:
            template = self.transform(template)
            search = self.transform(search)
        
        return {
            'template': template,
            'search': search,
            'template_bbox': torch.tensor(sample['template_bbox']),
            'search_bbox': torch.tensor(sample['search_bbox'])
        }

def get_lasot_dataloader(root_dir, batch_size=32, num_workers=2, split='train'):
    dataset = LaSOTDataset(root_dir=root_dir, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
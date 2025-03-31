import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import TrackTransformer
from data_loader import get_lasot_dataloader

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    with tqdm(total=num_batches, desc='Training') as pbar:
        for batch in dataloader:
            # 获取数据
            template = batch['template'].to(device)
            search = batch['search'].to(device)
            template_bbox = batch['template_bbox'].to(device)
            search_bbox = batch['search_bbox'].to(device)
            
            # 前向传播
            pred_bbox = model(search)
            loss = criterion(pred_bbox, search_bbox)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        with tqdm(total=num_batches, desc='Validation') as pbar:
            for batch in dataloader:
                template = batch['template'].to(device)
                search = batch['search'].to(device)
                template_bbox = batch['template_bbox'].to(device)
                search_bbox = batch['search_bbox'].to(device)
                
                pred_bbox = model(search)
                loss = criterion(pred_bbox, search_bbox)
                
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def main():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # 创建数据加载器
    train_loader = get_lasot_dataloader('data/lasot', batch_size=batch_size, split='train')
    val_loader = get_lasot_dataloader('data/lasot', batch_size=batch_size, split='val')
    
    # 创建模型
    model = TrackTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Training Loss: {train_loss:.4f}')
        
        # 验证阶段
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')

if __name__ == '__main__':
    main()
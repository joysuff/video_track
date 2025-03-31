# 基于Transformer的单流视频目标追踪算法

本项目实现了一个基于Transformer架构的单流视频目标追踪算法，使用LaSOT数据集的airplane类别进行训练和评估。

## 项目结构

- `data_loader.py`: 实现LaSOT数据集的加载和预处理
- `model.py`: 定义Transformer追踪器模型架构
- `train.py`: 实现模型训练和验证流程
- `evaluate.py`: 实现模型评估和结果可视化
- `requirements.txt`: 项目依赖包列表

## 环境配置

1. 创建并激活Python虚拟环境（推荐）

2. 安装依赖包：

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

1. 准备数据集

   - 下载LaSOT数据集
   - 将数据集放置在指定目录

2. 训练模型

   ```bash
   python train.py
   ```

3. 评估模型

   ```bash
   python evaluate.py
   ```

## 模型架构

- 基于Vision Transformer的编码器-解码器架构
- 使用多头自注意力机制处理视觉特征
- 输出目标边界框坐标 (x, y, w, h)

## 性能评估

- 使用IoU（交并比）作为评估指标
- 支持可视化追踪结果
- 提供详细的评估报告

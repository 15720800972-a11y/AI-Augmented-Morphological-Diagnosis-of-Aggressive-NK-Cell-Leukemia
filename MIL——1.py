import torch
import torch.nn as nn
import numpy as np
import re
import shutil
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
import os
from PIL import Image
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


# 配置参数设置（备注：权重模型定稿！）
def get_config():
    parser = argparse.ArgumentParser(description="病理图像分析系统")
    parser.add_argument("--data_root", type=str, default="C:/Users/bio-032/Desktop/NK",
                        help="数据根目录路径")
    parser.add_argument("--label_file", type=str, default="labels.csv",
                        help="标签文件名")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--input_size", type=int, default=224,
                        help="图像输入尺寸")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="训练轮次")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--save_dir", type=str, default="C:/Users/bio-032/Desktop//saved_models",
                        help="模型保存路径")
    parser.add_argument("--top_cells_dir", type=str, default="C:/Users/bio-032/Desktop/top_cells",
                        help="高权重细胞保存路径")
    parser.add_argument("--top_k", type=int, default=5,
                        help="每个病例保存的最高权重细胞数量")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--max_cells_per_patient", type=int, default=100,
                        help="每个病例处理的最大细胞数")
    return parser.parse_args()



# 医学影像数据集类


class MedicalDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.label_path = os.path.join(config.data_root, config.label_file)
        self.label_df = self._load_labels()
        self.patient_dirs = self._get_patient_paths()

        # 改进后的预处理流程
        self.transform = transforms.Compose([
            # 保持比例的缩放和填充
            transforms.Lambda(lambda img: self._preserve_ratio_resize(img)),

            # 颜色增强（病理图像适用参数）
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02
            ),

            # 空间增强
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15, fill=0),

            # 转换为张量
            transforms.ToTensor(),

            # 标准化（建议根据实际数据重新计算）
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

            # 随机擦除
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                value='random'
            )
        ])

    def _preserve_ratio_resize(self, img):
        """保持长宽比的缩放函数"""
        original_width, original_height = img.size
        target_size = self.config.input_size

        # 计算缩放比例
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 高质量缩放
        img = img.resize(
            (new_width, new_height),
            resample=Image.LANCZOS
        )

        # 创建黑色画布
        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

        # 居中粘贴
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        canvas.paste(img, (paste_x, paste_y))

        return canvas

    def _load_labels(self):
        df = pd.read_csv(self.label_path)
        assert {'patient_id', 'ANKL'}.issubset(df.columns), "CSV缺少必要列"
        return df

    def _get_patient_paths(self):
        return [os.path.join(self.config.data_root, f"patient_{row['patient_id']}")
                for _, row in self.label_df.iterrows()]

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = re.search(r'patient_(\d+)', os.path.basename(patient_dir)).group(1).zfill(3)

        img_files = sorted([f for f in os.listdir(patient_dir) if f.endswith(('.jpg', '.png'))])
        cells = []
        cell_paths = []

        for f in img_files[:self.config.max_cells_per_patient]:  # 建议配置化
            img_path = os.path.join(patient_dir, f)
            try:
                img = Image.open(img_path).convert('RGB')
                # 应用改进后的transform
                cells.append(self.transform(img))
                cell_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue

        return {
            'cells': cells,
            'label': torch.tensor(self.label_df.iloc[idx].ANKL, dtype=torch.float32),
            'patient_id': patient_id,
            'cell_paths': cell_paths
        }


# 数据整理函数
def mil_collate_fn(batch):
    """修正后的数据整理函数"""
    return {
        # 每个病例的细胞图像转换为张量 [num_cells, C, H, W]
        'cells': [torch.stack(item['cells']) for item in batch],  # 新增torch.stack
        'label': torch.stack([item['label'] for item in batch]),
        'patient_ids': [item['patient_id'] for item in batch],
        'cell_paths': [item['cell_paths'] for item in batch]
    }

# 改进的MIL模型（带注意力权重）
class SimplifiedMIL(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 注意力分支
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 分类器（移除最后的Sigmoid）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x_list):
        batch_preds = []
        batch_weights = []

        for cells in x_list:  # cells的形状: [num_cells, C, H, W]
            device = next(self.parameters()).device

            # 分块处理特征（直接处理张量）
            all_features = []
            num_cells = cells.size(0)

            for i in range(0, num_cells, 32):
                chunk = cells[i:i + 32].to(device)
                features = self.feature_extractor(chunk).squeeze()
                all_features.append(features)

            features = torch.cat(all_features)  # 形状: [num_cells, 512]

            # 计算注意力权重
            attn_weights = self.attention(features)  # [num_cells, 1]
            attn_weights = torch.softmax(attn_weights, dim=0)

            # 加权聚合
            aggregated = (features * attn_weights).sum(dim=0)  # [512]

            # 分类预测
            pred = self.classifier(aggregated)  # [1]

            batch_preds.append(pred)
            batch_weights.append(attn_weights.detach().cpu().numpy())

        return torch.stack(batch_preds).view(-1), batch_weights


# 保存高权重细胞
def save_top_cells(config, model, loader):
    model.eval()
    os.makedirs(config.top_cells_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            cells_list = [cells.to(config.device) for cells in batch['cells']]
            _, batch_weights = model(cells_list)

            for i, (weights, paths) in enumerate(zip(batch_weights, batch['cell_paths'])):
                patient_id = batch['patient_ids'][i]
                weights = weights.squeeze()

                # 获取top-k索引
                top_indices = np.argsort(weights)[-config.top_k:]

                # 创建病例目录
                patient_dir = os.path.join(config.top_cells_dir, f"patient_{patient_id}")
                os.makedirs(patient_dir, exist_ok=True)

                # 复制图像
                for idx in top_indices:
                    src_path = paths[idx]
                    dst_path = os.path.join(patient_dir, os.path.basename(src_path))
                    shutil.copy2(src_path, dst_path)

    print(f"\n✅ 高权重细胞已保存至: {os.path.abspath(config.top_cells_dir)}")


# 训练验证函数
def train_model(config):
    # 初始化
    device = torch.device(config.device)
    os.makedirs(config.save_dir, exist_ok=True)

    # 数据集准备
    full_dataset = MedicalDataset(config)

    # 分层划分数据集
    labels = [full_dataset[i]['label'].item() for i in range(len(full_dataset))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            collate_fn=mil_collate_fn)

    # 模型配置
    model = SimplifiedMIL().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()

    # 训练循环
    best_acc = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        train_preds, train_labels = [], []

        for batch in train_loader:
            cells_list = [cells.to(device) for cells in batch['cells']]
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(cells_list)
            loss = criterion(torch.sigmoid(outputs), labels)
            loss.backward()
            optimizer.step()

            train_preds.append(outputs.detach().cpu())
            train_labels.append(labels.cpu())

        # 计算训练指标
        train_preds = torch.cat(train_preds).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_loss = loss.item()
        train_acc = accuracy_score(train_labels, train_preds > 0.5)

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config)
            }, os.path.join(config.save_dir, 'best_model.pth'))

        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n")

    # 保存高权重细胞
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pth'))['model_state_dict'])
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, collate_fn=mil_collate_fn)
    save_top_cells(config, model, full_loader)

    return model


# 评估函数（简化版）
def evaluate(model, loader, device, criterion):
    model.eval()
    preds, labels, loss_list = [], [], []  # 添加loss_list初始化

    with torch.no_grad():
        for batch in loader:
            cells_list = [cells.to(device) for cells in batch['cells']]
            labels_batch = batch['label'].to(device)

            outputs, _ = model(cells_list)
            loss = criterion(torch.sigmoid(outputs), labels_batch)

            loss_list.append(loss.item())  # 收集每个batch的loss
            preds.append(outputs.detach().cpu())
            labels.append(labels_batch.cpu())

    # 计算平均损失
    avg_loss = np.mean(loss_list) if len(loss_list) > 0 else 0.0
    acc = accuracy_score(torch.cat(labels).numpy(), torch.cat(preds).numpy() > 0.5)
    return avg_loss, acc


if __name__ == "__main__":
    config = get_config()
    print("\n" + "=" * 50)
    print(f"🏥 病理分析系统启动")
    print(f"📂 数据路径: {os.path.abspath(config.data_root)}")
    print(f"💾 模型保存路径: {os.path.abspath(config.save_dir)}")
    print(f"🔝 高权重细胞保存路径: {os.path.abspath(config.top_cells_dir)}")
    print("=" * 50 + "\n")

    train_model(config)
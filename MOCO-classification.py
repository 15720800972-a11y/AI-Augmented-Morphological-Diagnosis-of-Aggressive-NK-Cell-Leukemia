import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import argparse


# ====== 1. MoCo模型定义 ======
class MoCo(nn.Module):
    def __init__(self, base_encoder='resnet50', dim=128):
        super().__init__()
        # 加载 ResNet50
        base = models.__dict__[base_encoder](pretrained=False)  # 推理时不一定要预训练参数，因为会加载checkpoint

        self.encoder_q = nn.Sequential(
            *list(base.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # 注意：推理阶段其实只需要 encoder_q，encoder_k 和 queue 可以不初始化以节省显存
        # 但为了保持加载权重时的结构一致性，这里保留定义，但设为不需要梯度

        # 投影头
        feature_dim = 2048  # ResNet50
        self.proj_q = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        # 推理模式下直接返回特征
        return self.extract_feat(x)

    def extract_feat(self, x):
        # 根据你的逻辑，使用投影后的特征进行比对
        feat = self.proj_q(self.encoder_q(x))
        feat = nn.functional.normalize(feat, dim=1)
        return feat


# ====== 2. 数据集类 ======
class ImgDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.img_paths = []
        if os.path.exists(root):
            self.img_paths = [os.path.join(root, f) for f in os.listdir(root)
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        else:
            print(f"Warning: Directory not found: {root}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        try:
            img = Image.open(p).convert('RGB')
            return self.transform(img), p
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return torch.zeros(3, 224, 224), p

    def __len__(self):
        return len(self.img_paths)


# ====== 3. 工具函数 ======
def extract_feats(model, dataloader, device):
    feats, paths = [], []
    model.eval()
    if len(dataloader) == 0:
        return np.array([]), []

    with torch.no_grad():
        for x, ps in tqdm(dataloader, desc="Extracting features"):
            x = x.to(device)
            f = model.extract_feat(x)  # shape [B, feat_dim]
            feats.append(f.cpu())
            paths += list(ps)

    if len(feats) > 0:
        feats = torch.cat(feats, dim=0).numpy()
    else:
        feats = np.array([])
    return feats, paths


def cosine_sim(v1, v2):
    # 加上 epsilon 防止除零
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


# ====== 4. 配置与主程序 ======
def get_args():
    parser = argparse.ArgumentParser(description="MoCo Centroid Evaluation")

    # 基础配置
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help='包含各类别子文件夹的根目录')
    parser.add_argument('--val_dir', type=str, default='./dataset/unknown_val',
                        help='待验证/待分类的文件夹路径')
    parser.add_argument('--checkpoint', type=str, default='./models/moco_encoder_q.pth',
                        help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.8, help='相似度阈值')

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 模型初始化
    print(f"Loading model from {args.checkpoint}...")
    model = MoCo(base_encoder='resnet50').to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        # 兼容性加载：处理可能存在的 'module.' 前缀或不匹配的键
        state_dict = ckpt.get('state_dict', ckpt)  # 尝试获取 state_dict
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Model loaded. Missing keys (safe to ignore if decoder keys): {len(msg.missing_keys)}")
    else:
        print("Error: Checkpoint file not found!")
        return

    model.eval()

    # 2. 定义参考库 (Prototypes)
    # 修改：不再硬编码A/B/C，而是自动扫描 data_root 下的子文件夹作为类别
    reference_dirs = {}
    if os.path.exists(args.data_root):
        for d in os.listdir(args.data_root):
            full_path = os.path.join(args.data_root, d)
            if os.path.isdir(full_path) and d != "unknown_val":  # 排除验证集
                reference_dirs[d] = full_path

    print(f"Found {len(reference_dirs)} reference classes: {list(reference_dirs.keys())}")

    # 3. 提取特征中心 (Centroids)
    centers = {}
    for class_name, dir_path in reference_dirs.items():
        print(f"\nProcessing Reference Class: {class_name}...")
        loader = DataLoader(ImgDataset(dir_path), args.batch_size, num_workers=2)
        feats, _ = extract_feats(model, loader, device)

        if len(feats) > 0:
            # 计算中心向量 (Mean Vector)
            center = feats.mean(axis=0)
            centers[class_name] = center
        else:
            print(f"Warning: Class {class_name} is empty!")

    # 4. 处理验证集
    print(f"\nProcessing Validation Set: {args.val_dir}...")
    val_loader = DataLoader(ImgDataset(args.val_dir), args.batch_size, num_workers=2)
    val_feats, val_paths = extract_feats(model, val_loader, device)

    if len(val_feats) == 0:
        print("Validation set is empty.")
        return

    # 5. 比对与分类
    print("\nCalculating similarities...")
    results = []

    for i, feat in enumerate(val_feats):
        path = val_paths[i]
        filename = os.path.basename(path)

        row = {'filename': filename, 'path': path}
        max_score = -1.0
        best_class = "unknown"

        # 计算与每个中心的距离
        for class_name, center in centers.items():
            sim = cosine_sim(feat, center)
            row[f'sim_{class_name}'] = sim

            if sim > max_score:
                max_score = sim
                best_class = class_name

        row['max_sim'] = max_score
        # 应用阈值判定
        row['final_pred'] = best_class if max_score > args.threshold else "unknown"
        results.append(row)

    # 6. 保存结果
    df = pd.DataFrame(results)
    output_csv = 'classification_results.csv'
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 任务完成！结果已保存至: {output_csv}")
    print("分类统计:")
    print(df['final_pred'].value_counts())


if __name__ == '__main__':
    main()
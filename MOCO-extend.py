import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import shutil
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse


# ====== 1. MoCo 模型定义 (修复版) ======
class MoCo(nn.Module):
    def __init__(self, base_encoder='resnet50', dim=128, K=8192, m=0.999, T=0.07):
        super().__init__()
        # 使用 ResNet50
        base = models.__dict__[base_encoder](weights="IMAGENET1K_V1")

        # Query Encoder (去除最后的全连接层)
        self.encoder_q = nn.Sequential(
            *list(base.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # Key Encoder
        self.encoder_k = nn.Sequential(
            *list(base.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            feature_dim = self.encoder_q(dummy).shape[1]

        # 投影头
        self.proj_q = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )
        self.proj_k = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

        self.m = m
        self.T = T
        self.K = K

        # 初始化 Key Encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # 训练时的前向传播
    def forward(self, im_q, im_k):
        q = self.proj_q(self.encoder_q(im_q))
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.proj_k(self.encoder_k(im_k))
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(im_q.device)

        # 简单更新队列 (简化版)
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = k.T
            ptr = (ptr + batch_size) % self.K
        else:
            self.queue[:, :-batch_size] = self.queue[:, batch_size:].clone()
            self.queue[:, -batch_size:] = k.T.detach()
            ptr = 0
        self.queue_ptr[0] = ptr

        return nn.CrossEntropyLoss()(logits, labels)

    # 新增：专门用于推理解压特征的函数
    def extract_features(self, x):
        feat = self.encoder_q(x)
        return F.normalize(feat, dim=1)


# ====== 2. 可视化类 (修复重复定义) ======
class VisualEvaluator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set(style="whitegrid")

    def plot_tsne(self, features, labels, title="t-SNE Visualization"):
        plt.figure(figsize=(10, 8))
        print("正在进行 PCA 降维...")
        pca = PCA(n_components=min(50, features.shape[1]))
        reduced_features = pca.fit_transform(features)

        print("正在进行 t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embeddings = tsne.fit_transform(reduced_features)

        unique_labels = list(set(labels))
        colors = sns.color_palette("Set2", n_colors=len(unique_labels))

        for i, label in enumerate(unique_labels):
            idxs = [idx for idx, l in enumerate(labels) if l == label]
            label_name = "Unlabeled Pool" if label == 0 else "Annotated (Query)"
            plt.scatter(
                embeddings[idxs, 0], embeddings[idxs, 1],
                color=colors[i], label=label_name, alpha=0.6, s=15
            )

        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, "tsne_visualization.png"), dpi=300)
        plt.close()

    def plot_similarity_distribution(self, similarities, threshold):
        plt.figure(figsize=(10, 6))
        sns.histplot(similarities, bins=50, kde=True, color="royalblue")
        plt.axvline(threshold, color='tomato', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "similarity_dist.png"), dpi=300)
        plt.close()

    def plot_retrieval_examples(self, query_img, result_imgs, idx):
        # 确保 result_imgs 不超过5张
        current_results = result_imgs[:5]
        cols = len(current_results) + 1

        fig, ax = plt.subplots(1, cols, figsize=(3 * cols, 3))

        # 画 Query
        ax[0].imshow(query_img.resize((224, 224)))
        ax[0].axis('off')
        ax[0].set_title("Query Image", fontsize=10, color='blue')

        # 画 Results
        for i, img in enumerate(current_results):
            ax[i + 1].imshow(img.resize((224, 224)))
            ax[i + 1].axis('off')
            ax[i + 1].set_title(f"Top {i + 1}", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"retrieval_group_{idx}.png"), dpi=300)
        plt.close()


# ====== 3. 特征提取器 (修复逻辑错误) ======
class FeatureExtractor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # 初始化模型
        self.model = MoCo(base_encoder='resnet50', dim=128).to(self.device)

        # 加载权重
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # 处理 state_dict 键名可能带 'module.' 前缀的问题
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # 如果 MoCo 存的时候只有 encoder_q，需按需加载
            # 这里假设加载完整 MoCo 权重，稍作宽容处理
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning loading weights: {e}")
        else:
            print("Warning: Checkpoint not found, using random weights!")

        self.model.eval()

    def extract(self, dataloader):
        features = []
        paths = []
        print("开始提取特征...")
        with torch.no_grad():
            for i, (images, batch_paths) in enumerate(dataloader):
                images = images.to(self.device)
                # !!! 关键修正：调用 extract_features 而不是 forward !!!
                feats = self.model.extract_features(images)
                features.append(feats.cpu())
                paths.extend(batch_paths)
                if i % 10 == 0:
                    print(f"Processed batch {i}/{len(dataloader)}")
        return torch.cat(features, dim=0), paths


# ====== 4. 数据集定义 ======
class SearchDataset(Dataset):
    def __init__(self, root):
        self.img_paths = [os.path.join(root, f) for f in os.listdir(root)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except:
            # 容错处理
            return torch.zeros(3, 224, 224), path

    def __len__(self):
        return len(self.img_paths)


# ====== 5. 主配置与流程 (修复路径) ======
class SearchConfig:
    # 路径改为相对路径
    data_path = "./unlabeled_pool"  # 未标注的大池子
    annotated_dir = "./annotated_query"  # 已标注的种子数据（用来做Query）
    output_dir = "./retrieval_results"  # 结果输出
    checkpoint_path = "./saved_models/moco_best.pth"  # MoCo 权重


def find_similar_images(config):
    # 0. 准备工作
    os.makedirs(config.output_dir, exist_ok=True)
    extractor = FeatureExtractor(config.checkpoint_path)

    # 1. 提取未标注池子的特征
    print(f"Loading Unlabeled Pool: {config.data_path}")
    full_dataset = SearchDataset(config.data_path)
    if len(full_dataset) == 0:
        print("Error: Unlabeled dataset is empty.")
        return
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=0)
    all_features, all_paths = extractor.extract(full_loader)

    # 2. 提取种子（Query）图片的特征
    print(f"Loading Query Set: {config.annotated_dir}")
    annotated_dataset = SearchDataset(config.annotated_dir)
    if len(annotated_dataset) == 0:
        print("Error: Query dataset is empty.")
        return
    annotated_loader = DataLoader(annotated_dataset, batch_size=64, shuffle=False)
    anno_features, anno_paths = extractor.extract(annotated_loader)

    # 3. 计算相似度矩阵 (Cosine Similarity)
    # 因为我们在 extract_features 里已经做了 l2 normalize，所以直接矩阵相乘就是 cosine similarity
    print("Calculating Similarity Matrix...")
    sim_matrix = torch.mm(all_features, anno_features.T)  # [N_pool, N_query]

    # 找到每张未标注图片与所有 Query 图片的最大相似度
    max_sim, _ = torch.max(sim_matrix, dim=1)

    # 4. 设定阈值筛选
    threshold = torch.mean(max_sim) + 1.5 * torch.std(max_sim)  # 稍微严格一点 (+1.5 std)
    print(f"Threshold set to: {threshold:.4f}")

    # 5. 保存结果
    results = []
    anno_filenames = set([os.path.basename(p) for p in anno_paths])

    for path, score in zip(all_paths, max_sim.numpy()):
        filename = os.path.basename(path)
        # 排除掉已经是种子集里的图
        if score >= threshold and filename not in anno_filenames:
            dest = os.path.join(config.output_dir, filename)
            shutil.copy(path, dest)
            results.append((filename, score))

    # 写入 CSV
    with open(os.path.join(config.output_dir, 'results.csv'), 'w') as f:
        f.write("filename,similarity_score\n")
        for name, score in sorted(results, key=lambda x: -x[1]):
            f.write(f"{name},{score:.4f}\n")

    # 6. 可视化
    visualizer = VisualEvaluator(config.output_dir)

    # 6.1 t-SNE
    try:
        # 下采样以加快画图
        sample_size = min(1000, len(all_features))
        if sample_size > 0:
            sample_idx = np.random.choice(len(all_features), sample_size, replace=False)
            combined_features = torch.cat([all_features[sample_idx], anno_features]).numpy()
            # 0 代表未标注, 1 代表 Query
            labels = [0] * len(sample_idx) + [1] * len(anno_features)
            visualizer.plot_tsne(combined_features, labels)
    except Exception as e:
        print(f"t-SNE Error: {str(e)}")

    # 6.2 相似度分布
    visualizer.plot_similarity_distribution(max_sim.numpy(), threshold.item())

    # 6.3 检索图示 (取前3个Query展示)
    try:
        num_queries = min(3, len(anno_paths))
        # 对每一个 query 找最相似的 pool images
        for q_idx in range(num_queries):
            # 获取当前 query 对所有 pool 的相似度
            cur_sims = sim_matrix[:, q_idx]
            # 排序
            top_indices = torch.argsort(cur_sims, descending=True)[:5]

            query_img = Image.open(anno_paths[q_idx]).convert('RGB')
            result_imgs = [Image.open(all_paths[i]) for i in top_indices]

            visualizer.plot_retrieval_examples(query_img, result_imgs, q_idx)
    except Exception as e:
        print(f"Retrieval Vis Error: {str(e)}")

    print(f"\n✅ 完成！找到 {len(results)} 张相似图像。")
    print(f"📂 结果保存在: {os.path.abspath(config.output_dir)}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    find_similar_images(SearchConfig())
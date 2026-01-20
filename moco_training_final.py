import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image

# ====== MoCo模型定义 ======
class MoCo(nn.Module):
    def __init__(self, base_encoder='resnet50'):  # 由'resnet18'改为'resnet50'
        super().__init__()
        base = models.__dict__[base_encoder](weights="IMAGENET1K_V1")

        self.encoder_q = nn.Sequential(
            *list(base.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.encoder_k = nn.Sequential(
            *list(base.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 此处feature_dim应为2048
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            feature_dim = self.encoder_q(dummy).shape[1]
            assert feature_dim == 2048, f"特征维度应为2048，实际为{feature_dim}"

        # proj层输入维和输出维都需与ResNet50对应（2048维）
        self.proj_q = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.proj_k = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.m = 0.999
        device = next(self.parameters()).device
        self.queue_size = 8192
        self.register_buffer('queue',
                             nn.functional.normalize(
                                 torch.randn(128, self.queue_size, device=device),
                                 dim=0
                             ))

        for param in self.encoder_k.parameters():
            param.requires_grad_(False)
    ...

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, im_q, im_k):
        q = self.proj_q(self.encoder_q(im_q))
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = self.proj_k(self.encoder_k(im_k))
            k = nn.functional.normalize(k, dim=1)

        # 动态获取当前batch大小
        batch_size = k.size(0)

        # 对比损失计算
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(im_q.device)
        loss = nn.CrossEntropyLoss()(logits / 0.07, labels)

        # 安全更新队列
        with torch.no_grad():
            self.queue[:, :-batch_size] = self.queue[:, batch_size:].clone()
            self.queue[:, -batch_size:] = k.T.detach()

        return loss


class MedicalDataset(Dataset):
    def __init__(self, root):
        self.img_paths = [os.path.join(root, f) for f in os.listdir(root)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),  # 缩小min-scale,更强增强
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=360),  # 允许任意角度旋转
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # 色彩扰动略增强
            transforms.RandomGrayscale(p=0.1),  # 加灰度扰动
            transforms.GaussianBlur(kernel_size=5),  # 防过拟合细节
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img), self.transform(img)

    def __len__(self):
        return len(self.img_paths)

# ====== 配置类（建议在主脚本自定义继续复用）======
class Config:
    """
    配置基类（实际工程中可在主流程继续继承或覆盖参数）
    """
    data_path = "C:/AI/output"
    pretrained_path = "C:/AI/moco_224_epoch_90.pth"
    epochs = 500
    batch_size = 64
    num_workers = 4
    lr = 1e-5
    save_dir = "C:/AI/continued_models"
    save_interval = 5
    grad_accumulate = 2
    extract_feature = True
    eval_knn = False
    eval_umap = True
    knn_topk = 5
    save_feat_path = "all_features.npy"
    save_knn_csv = "sim_query_topk.csv"
    save_umap_png = "embedding_umap.png"
    # 可按需要在主逻辑补充如 all_labels = [...]
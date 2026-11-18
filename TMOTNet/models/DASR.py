import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Difficulty-Aware Semantic Regularization Module

#Difficult Sample Mining
class DSM(nn.Module):
    def __init__(self, kernel_size=3, beta=0.7):
        super().__init__()
        self.kernel_size = kernel_size
        self.beta = float(beta)

    def forward(self, change_logits):

        logits = change_logits  # [B,1,Hc,Wc]

        prob = torch.sigmoid(logits)                   # [B,1,H,W]
        prob = torch.clamp(prob, 1e-6, 1-1e-6)        # 防止 log nan

        entropy = -(prob * torch.log(prob) +
                    (1 - prob) * torch.log(1 - prob))  # [B,1,H,W]

        mean_entropy = F.avg_pool2d(
            entropy,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )  # [B,1,H,W]

        B, _, H, W = mean_entropy.shape
        ent_flat = mean_entropy.view(B, -1)           # [B, H*W]
        ent_min = ent_flat.min(dim=1)[0].view(B, 1, 1, 1)
        ent_max = ent_flat.max(dim=1)[0].view(B, 1, 1, 1)
        mask = (mean_entropy - ent_min) / (ent_max - ent_min + 1e-6)
        weight = self.beta + (1 - self.beta) * mask
        return weight

class WeightedChangeSimilarity(nn.Module):
    def __init__(self, eps=1e-6, detach_weight=True):
        super().__init__()
        self.eps = eps
        self.detach_weight = detach_weight
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='none')

    def forward(self, x1, x2, label_change, weight_mask=None):
        B, C, H, W = x1.shape

        # semantic probability
        p1 = F.softmax(x1, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
        p2 = F.softmax(x2, dim=1).permute(0, 2, 3, 1).reshape(-1, C)

        # target: unchanged +1, changed -1
        y = label_change.view(-1).float()
        target = 1.0 - 2.0 * (y > 0.5).float()

        # cosine embedding loss
        loss = self.loss_f(p1, p2, target)  # [N]

        # DSM weighting
        if weight_mask is not None:
            w = weight_mask.view(-1).float()
            if self.detach_weight:
                w = w.detach()
            return (loss * w).sum() / (w.sum() + self.eps)

        return loss.mean()
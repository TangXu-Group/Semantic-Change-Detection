import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ============================== Batched Sinkhorn ==============================

class SinkhornOT(nn.Module):
    """
    Batched balanced Sinkhorn OT.
    Inputs:
      cost: [B, N, L]  cost matrix (image tokens × text tokens)
      mu:   [B, N]     row marginals (optional; default uniform)
      nu:   [B, L]     col marginals (optional; default uniform)
    Returns:
      P:       [B, N, L]  transport plan
      ot_loss: [B]        sum(P * cost) per sample
    """
    def __init__(self, epsilon: float = 0.15, n_iters: int = 50, eps: float = 1e-8):
        super().__init__()
        self.epsilon = float(epsilon)
        self.n_iters = int(n_iters)
        self.eps = float(eps)

    def forward(self, cost: torch.Tensor, mu: Optional[torch.Tensor] = None, nu: Optional[torch.Tensor] = None):
        assert cost.dim() == 3, "cost must be [B, N, L]"
        B, N, L = cost.shape
        device, dtype = cost.device, cost.dtype

        if mu is None:
            mu = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
        if nu is None:
            nu = torch.full((B, L), 1.0 / L, device=device, dtype=dtype)

        K = torch.exp(-cost / self.epsilon).clamp_min(1e-12)  # [B, N, L]

        u = torch.full_like(mu, 1.0 / max(N, 1))
        v = torch.full_like(nu, 1.0 / max(L, 1))

        for _ in range(self.n_iters):
            Kv  = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)                 # [B, N]
            u   = mu / (Kv + self.eps)
            KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) # [B, L]
            v   = nu / (KTu + self.eps)

        P = u.unsqueeze(-1) * K * v.unsqueeze(1)                            # [B, N, L]
        P = P / (P.sum(dim=(1, 2), keepdim=True) + self.eps)                # mild renorm
        ot_loss = torch.sum(P * cost, dim=(1, 2))
        return P, ot_loss


# ===================== Marginals ====================

def build_marginals_with_change(
    sim_blN: torch.Tensor,                 # [B, L, N]  文本×图像 相似度 (cos)
    hw: Tuple[int, int],                   # (H, W) 其中 N=H*W
    change_prob: Optional[torch.Tensor] = None,  # [B,1,H,W] 或 [B,H,W]，尺寸需与 hw 一致
    lambda_unchg: float = 0.15,               # 行边际：变化先验权重
    lambda_t: float = 0.5,                 # 列边际：类别存在先验权重
    temp_t: float = 4.0,                   # 列边际 softmax 温度
    eps: float = 1e-8
):
    B, L, N = sim_blN.shape
    H, W = hw
    assert N == H * W, f"N={N} must equal H*W={H*W}"
    device, dtype = sim_blN.device, sim_blN.dtype

    # ----------------- Rows (mu): uniform ⊕ change prior -----------------
    mu_u = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)

    if change_prob is not None:
        cp = change_prob
        if cp.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            cp = cp.unsqueeze(1)
        assert cp.shape[-2:] == (H, W), f"change_prob size {tuple(cp.shape[-2:])} != hw={(H,W)}"

        unchg = (1-cp).detach()                    # [B,1,H,W]
        unchg_vec = unchg.flatten(1)             # [B, N]
        s = unchg_vec.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mu_unchg = unchg_vec / s                 # [B, N]

        # 混合
        mu = (1 - lambda_unchg) * mu_u + lambda_unchg * mu_unchg
    else:
        mu = mu_u

    mu = mu.clamp_min(eps)
    mu = mu / (mu.sum(dim=1, keepdim=True) + eps)
    mu = mu.detach()                          

    # ----------------- Cols (nu): uniform ⊕ class-presence prior -----------------
    nu_u = torch.full((B, L), 1.0 / L, device=device, dtype=dtype)
    score = sim_blN.detach().max(dim=2).values          # [B,L]  每类对最强图像 token 的响应
    logits = temp_t * (score - score.max(dim=1, keepdim=True).values)  # 数值稳定
    p = F.softmax(logits, dim=1)
    nu = (1 - lambda_t) * nu_u + lambda_t * p

    nu = nu.clamp_min(eps)
    nu = nu / (nu.sum(dim=1, keepdim=True) + eps)
    nu = nu.detach()

    return mu, nu


# ================================ Main module ================================

class OTAlignWithLoss(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        epsilon: float = 0.15,
        sinkhorn_iters: int = 50,
        lambda_unchg: float = 0.15,
        lambda_t: float = 0.5,
        temp_t: float = 4.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.C = in_channels
        self.D = embed_dim
        self.eps = float(eps)

        # E-space projections
        self.img_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.txt_in = nn.Linear(in_channels, embed_dim, bias=False)

        # Back to C-space
        self.img_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)
        
        self.alpha_img = nn.Parameter(torch.tensor(0.0))

        self.ot = SinkhornOT(epsilon=epsilon, n_iters=sinkhorn_iters, eps=eps)

        self.lambda_unchg = float(lambda_unchg)
        self.lambda_t   = float(lambda_t)
        self.temp_t     = float(temp_t)

    @staticmethod
    def _flatten_hw(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, D, H, W = x.shape
        y = x.view(B, D, H * W).transpose(1, 2).contiguous()  # [B, N, D]
        return y, (H, W)

    @staticmethod
    def _unflatten_hw(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        B, N, D = x.shape
        H, W = hw
        return x.transpose(1, 2).contiguous().view(B, D, H, W)

    def forward(
        self,
        img_feats: torch.Tensor,                # [B, C, H, W]
        txt_feats: torch.Tensor,                # [B, L, C]
        change_prob: Optional[torch.Tensor] = None,  # [B,1,H,W] or [B,H,W] SAME hw
        return_loss: bool = True,
    ):
        B, C, H, W = img_feats.shape

        # 1) project to common space (E)
        I_e = self.img_in(img_feats)  # [B, E, H, W]
        T_e = self.txt_in(txt_feats)  # [B, L, E]

        # 2) flatten image tokens
        I_flat, hw = self._flatten_hw(I_e)      # [B, N, E]

        # 3) cosine similarity & cost
        T_norm = F.normalize(T_e, dim=-1)
        I_norm = F.normalize(I_flat, dim=-1)
        sim  = torch.einsum("bld,bnd->bln", T_norm, I_norm)   # [B, L, N]
        cost = (1.0 - sim).transpose(1, 2).contiguous()       # [B, N, L]

        # 4) marginals
        mu, nu = build_marginals_with_change(
            sim_blN=sim, hw=hw, change_prob=change_prob,
            lambda_unchg=self.lambda_unchg,
            lambda_t=self.lambda_t, temp_t=self.temp_t, eps=self.eps
        )

        # 5) Sinkhorn OT
        P, ot_loss = self.ot(cost, mu=mu, nu=nu)

        # 6) transport-weighted aggregation (in E-space)
        ET = torch.bmm(P, T_e)                      # [B, N, E]
        EI = torch.bmm(P.transpose(1, 2), I_flat)   # [B, L, E]
        
        # 7) 残差融合
        ET_map = self._unflatten_hw(ET, hw)         # [B, E, H, W]
        ET_c   = self.img_out(ET_map)               # [B, C, H, W]
        img_aligned = img_feats + torch.tanh(self.alpha_img) * ET_c

        if return_loss:
            return img_aligned, ot_loss.mean()
        else:
            return img_aligned, None

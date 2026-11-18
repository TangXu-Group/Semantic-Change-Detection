import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import initialize_weights, init_deconv_with_bilinear


class PCASE(nn.Module):
    """
    Progressive Change-Aware Semantic Enhancement (PCASE)

    输入:
        feats1: [x11, x12, x13, x14]  (T1)
        feats2: [x21, x22, x23, x24]  (T2)

    输出:
        out1_1, out1_2
        若 return_logits=True:
        额外返回 (logits_1, logits_2, logits_3)
    """

    def __init__(
        self,
        in_channels=[64, 128, 128, 128],
        out_channels: int = 128,
        return_logits: bool = False,
    ):
        super().__init__()

        c1, c2, c3, c4 = in_channels
        self.out_channels = int(out_channels)
        self.return_logits = bool(return_logits)

        # ---------------- upsample ----------------
        self.up_4_to_3 = nn.ConvTranspose2d(c4, c4, 2, 2, bias=False)
        self.up_3_to_2 = nn.ConvTranspose2d(out_channels, out_channels, 2, 2, bias=False)
        self.up_2_to_1 = nn.ConvTranspose2d(out_channels, out_channels, 2, 2, bias=False)

        # ================= level 3 =================
        self.local3 = nn.Sequential(
            nn.Conv2d(c3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse3 = nn.Sequential(
            nn.Conv2d(c3 + c4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.global3 = nn.Conv2d(2 * c4, out_channels, 1, bias=False)

        self.mask3 = nn.Conv2d(out_channels, 1, 1, bias=False)

        # ================= level 2 =================
        self.local2 = nn.Sequential(
            nn.Conv2d(c2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse2 = nn.Sequential(
            nn.Conv2d(c2 + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.global2 = nn.Conv2d(2 * out_channels, out_channels, 1, bias=False)

        self.mask2 = nn.Conv2d(out_channels, 1, 1, bias=False)

        # ================= level 1 =================
        self.local1 = nn.Sequential(
            nn.Conv2d(c1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(c1 + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.global1 = nn.Conv2d(2 * out_channels, out_channels, 1, bias=False)

        self.mask1 = nn.Conv2d(out_channels, 1, 1, bias=False)

        # residual gates
        self.gamma3 = nn.Parameter(torch.tensor(0.0))
        self.gamma2 = nn.Parameter(torch.tensor(0.0))
        self.gamma1 = nn.Parameter(torch.tensor(0.0))

        # ---------------- initialization ----------------
        init_deconv_with_bilinear(self.up_4_to_3)
        init_deconv_with_bilinear(self.up_3_to_2)
        init_deconv_with_bilinear(self.up_2_to_1)

        initialize_weights(
            self.local3, self.fuse3, self.global3, self.mask3,
            self.local2, self.fuse2, self.global2, self.mask2,
            self.local1, self.fuse1, self.global1, self.mask1,
        )

    def forward(self, feats1, feats2):

        x11, x12, x13, x14 = feats1
        x21, x22, x23, x24 = feats2

        # ======================================================
        # level 3
        # ======================================================
        x14_up = self.up_4_to_3(x14) if x14.size()[2:] != x13.size()[2:] else x14
        x24_up = self.up_4_to_3(x24) if x24.size()[2:] != x23.size()[2:] else x24

        L3_1 = self.local3(x13)
        L3_2 = self.local3(x23)

        F3_1 = self.fuse3(torch.cat([x13, x14_up], dim=1))
        F3_2 = self.fuse3(torch.cat([x23, x24_up], dim=1))

        diff3 = torch.abs(F3_1 - F3_2)
        logits_3 = self.mask3(diff3)
        mask3 = torch.sigmoid(logits_3)

        stable3_1 = x14_up * (1.0 - mask3)
        stable3_2 = x24_up * (1.0 - mask3)

        g3_shared = self.global3(torch.cat([stable3_1, stable3_2], dim=1))

        S3_1 = g3_shared + L3_1 * mask3
        S3_2 = g3_shared + L3_2 * mask3

        g3 = torch.tanh(self.gamma3)

        out3_1 = F3_1 + g3 * S3_1
        out3_2 = F3_2 + g3 * S3_2

        # ======================================================
        # level 2
        # ======================================================
        out3_1_up = self.up_3_to_2(out3_1)
        out3_2_up = self.up_3_to_2(out3_2)

        L2_1 = self.local2(x12)
        L2_2 = self.local2(x22)

        F2_1 = self.fuse2(torch.cat([x12, out3_1_up], dim=1))
        F2_2 = self.fuse2(torch.cat([x22, out3_2_up], dim=1))

        diff2 = torch.abs(F2_1 - F2_2)
        logits_2 = self.mask2(diff2)
        mask2 = torch.sigmoid(logits_2)

        stable2_1 = out3_1_up * (1.0 - mask2)
        stable2_2 = out3_2_up * (1.0 - mask2)

        g2_shared = self.global2(torch.cat([stable2_1, stable2_2], dim=1))

        S2_1 = g2_shared + L2_1 * mask2
        S2_2 = g2_shared + L2_2 * mask2

        g2 = torch.tanh(self.gamma2)

        out2_1 = F2_1 + g2 * S2_1
        out2_2 = F2_2 + g2 * S2_2

        # ======================================================
        # level 1
        # ======================================================
        out2_1_up = self.up_2_to_1(out2_1)
        out2_2_up = self.up_2_to_1(out2_2)

        L1_1 = self.local1(x11)
        L1_2 = self.local1(x21)

        F1_1 = self.fuse1(torch.cat([x11, out2_1_up], dim=1))
        F1_2 = self.fuse1(torch.cat([x21, out2_2_up], dim=1))

        diff1 = torch.abs(F1_1 - F1_2)
        logits_1 = self.mask1(diff1)
        mask1 = torch.sigmoid(logits_1)

        stable1_1 = out2_1_up * (1.0 - mask1)
        stable1_2 = out2_2_up * (1.0 - mask1)

        g1_shared = self.global1(torch.cat([stable1_1, stable1_2], dim=1))

        S1_1 = g1_shared + L1_1 * mask1
        S1_2 = g1_shared + L1_2 * mask1

        g1 = torch.tanh(self.gamma1)

        out1_1 = F1_1 + g1 * S1_1
        out1_2 = F1_2 + g1 * S1_2

        if self.return_logits:
            return out1_1, out1_2, (logits_1, logits_2, logits_3)

        return out1_1, out1_2
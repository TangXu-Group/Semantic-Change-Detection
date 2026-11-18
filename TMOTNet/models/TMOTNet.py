import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import models
from utils.misc import initialize_weights, init_deconv_with_bilinear
from .OT import OTAlignWithLoss
from .PCASE import PCASE


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        self.head3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self.head3, self.head4)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class TMOTNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, classes=None, device='cuda'):
        super().__init__()
        self.device = device
        self.classes = classes

        # 缓存文本特征
        self._cached_text_feats = None

        # ----- backbone -----
        self.FCN = FCN(in_channels, pretrained=True)

        # ----- decoder -----
        self.Dec = PCASE(in_channels=[64, 128, 128, 128], out_channels=128)

        # ----- heads -----
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.resCD = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        # change prior head
        self.change_prior_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # ----- CLIP text encoder + adapter -----
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        self.text_adapter = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        ).to(device)

        # ----- OT aligner -----
        self.OTAlignWithLoss = OTAlignWithLoss(
            in_channels=128, embed_dim=256, epsilon=0.15,
            sinkhorn_iters=50, lambda_unchg=0.5, lambda_t=1.0, temp_t=2.0
        )
        # init heads
        initialize_weights(self.classifier1, self.classifier2, self.resCD, self.classifierCD, self.change_prior_head)

    # ---------- text encoding ----------
    def encode_text(self):
        assert self.classes is not None and len(self.classes) > 0, "classes list must be provided"

        if self._cached_text_feats is not None:
            return self._cached_text_feats

        prompts = [f"a satellite image of {cls}" for cls in self.classes]
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)  # [L, 512]
            text_feats = F.normalize(text_feats, dim=-1)
    
        self._cached_text_feats = text_feats
        return self._cached_text_feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes)
            )
        layers = [block(inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def base_forward(self, x):
        x = self.FCN.layer0(x)
        x = self.FCN.maxpool(x)
        x1 = self.FCN.layer1(x)  # 1/4
        x2 = self.FCN.layer2(x1) # 1/8
        x3 = self.FCN.layer3(x2) # 1/16
        x4 = self.FCN.layer4(x3) # 1/16
        x3 = self.FCN.head3(x3)  # [B,128,H/16,W/16]
        x4 = self.FCN.head4(x4)  # [B,128,H/16,W/16]
        return x1, x2, x3, x4

    # ---------- forward ----------
    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        x_size = t1.size()

        # backbone
        x11, x12, x13, x14 = self.base_forward(t1)
        x21, x22, x23, x24 = self.base_forward(t2)

        # text feats
        text_feats = self.text_adapter(self.encode_text())  # [L, 128]
        B = t1.size(0)
        text_feats_batch = text_feats.unsqueeze(0).expand(B, -1, -1)  # [B, L, 128]

        # ----- change_prob -----
        change_logits = self.change_prior_head(torch.abs(x14 - x24))     # [B,1,H/16,W/16]
        change_prob = torch.sigmoid(change_logits)       # [B,1,H/16,W/16], 作为 OT 行边际先验

        # ----- OT alignment -----
        x14_aligned, ot_loss1 = self.OTAlignWithLoss(
            img_feats=x14, txt_feats=text_feats_batch,
            change_prob=change_prob, return_loss=True
        )
        x24_aligned, ot_loss2 = self.OTAlignWithLoss(
            img_feats=x24, txt_feats=text_feats_batch,
            change_prob=change_prob, return_loss=True
        )
        ot_loss = 0.5 * (ot_loss1 + ot_loss2)

        # ----- PCASE decoder -----
        x1_feat, x2_feat = self.Dec(
            [x11, x12, x13, x14_aligned],
            [x21, x22, x23, x24_aligned]
        )

        # CD branch    
        diff = torch.abs(x1_feat - x2_feat)
        fuse = self.fuse(torch.cat([x1_feat, x2_feat, diff], dim=1))
        xc = self.resCD(fuse)                                      
        change = self.classifierCD(xc)

        out1 = self.classifier1(x1_feat)
        out2 = self.classifier2(x2_feat)

        change_map = F.interpolate(change, x_size[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, x_size[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, x_size[2:], mode='bilinear', align_corners=False)

        return change_map, out1, out2, change_logits, ot_loss

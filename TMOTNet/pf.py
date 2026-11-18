import torch
from thop import profile
from models.TMOTNet import TMOTNet
from datasets import RS_SECOND as RS

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = TMOTNet(in_channels=3, num_classes=7, classes=RS.CLASSES, device='cuda').to(device)
model.float()
model.eval()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"=== 参数统计 ===")
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"可学习参数量: {trainable_params / 1e6:.2f} M")
    print(f"冻结参数量: {frozen_params / 1e6:.2f} M")
    print(f"可学习参数比例: {trainable_params/total_params*100:.1f}%")
    
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)

B, C, H, W = 1, 3, 512, 512
dummy_input1 = torch.randn(B, C, H, W).to(device)
dummy_input2 = torch.randn(B, C, H, W).to(device)

flops, params = profile(model, inputs=(dummy_input1, dummy_input2))
print(f"\n=== FLOPs 分析 ===")
print(f"FLOPs: {flops / 1e9:.2f} G")
print(f"参数量 (thop): {params / 1e6:.2f} M")
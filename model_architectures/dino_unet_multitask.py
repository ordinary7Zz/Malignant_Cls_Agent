
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
    

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4)

        self.fuse = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn(self.fuse(x)))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)

    
class DINOv3_S_UNet_MULTITASK(nn.Module):
    def __init__(self, pretrained=True, use_dilation=False) -> None:
        super(DINOv3_S_UNet_MULTITASK, self).__init__()

        self.use_dilation = use_dilation

        self.dino = timm.create_model(model_name="vit_small_patch16_dinov3.lvd1689m",
            features_only=True,
            pretrained=pretrained,
        )
        # 获取DINO模型输出特征的通道数
        dino_channels = 384

        self.reduce1 = nn.Conv2d(dino_channels, 128, 1)
        self.reduce2 = nn.Conv2d(dino_channels, 128, 1)
        self.reduce3 = nn.Conv2d(dino_channels, 128, 1)
        self.reduce4 = nn.Conv2d(dino_channels, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

        # 根据参数决定是否使用Dilation层
        if self.use_dilation:
            self.dilate = DilatedConvBlock(128, 128)
        
        # ========= 分类任务结构 =========
        # 1. 增强的分类特征提取
        # 全局平均池化 + 全局最大池化 (GAP + GMP)，捕获不同类型的全局特征
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 2. 层注意力机制 - 学习不同DINO层特征的重要性
        self.layer_attention = nn.Sequential(
            nn.Linear(dino_channels * 3, 3),  # 3层特征
            nn.Softmax(dim=1)
        )
        
        # 3. 特征注意力机制 - 帮助模型关注重要特征
        self.feature_attention = nn.Sequential(
            nn.Linear(dino_channels, dino_channels),
            nn.Sigmoid()
        )
        
        # 4. 分类头设计
        # 使用所有3层特征的组合
        classification_feature_dim = dino_channels * 2  # GAP(384) + GMP(384)
        
        # 增强的分类特征提取器
        self.classification_backbone = nn.Sequential(
            nn.Linear(classification_feature_dim, 512),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 单独的任务头
        self.benign_malignant_head = nn.Linear(256, 1)  # 良恶性二分类
        self.tirads_head = nn.Linear(256, 5)  # TIRADS 5分类

    def forward(self, x):
        B, C, H, W = x.shape
        
        # ========= 分割任务 =========
        # 获取DINO的多个层次特征
        # [feat1, feat2, feat3] [B, 384, H, W]
        all_features = self.dino(x)  # 获取所有层次的特征 
        features = all_features[-1]  # 最后一层特征用于分割任务
        
        x1 = F.interpolate(self.reduce1(features), size=(H//4, W//4), mode='bilinear')
        x2 = F.interpolate(self.reduce2(features), size=(H//8, W//8), mode='bilinear')
        x3 = F.interpolate(self.reduce3(features), size=(H//16, W//16), mode='bilinear')
        x4 = F.interpolate(self.reduce4(features), size=(H//32, W//32), mode='bilinear')
        
        # 如果启用dilation,则在x4上应用DilatedConvBlock
        if self.use_dilation:
            x4 = self.dilate(x4)
        
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = F.interpolate(self.head(x), scale_factor=2, mode='bilinear')
        
        # ========= 分类任务 =========
        # 使用所有DINO特征层进行分类
        
        # 对所有特征层进行全局池化
        all_avg_features = []
        all_max_features = []
        
        # 从DINOv3模型的多层特征图中提取全局特征
        for feat in all_features:
            avg = self.global_avg_pool(feat).view(B, -1) # [B, 384]
            max_feat = self.global_max_pool(feat).view(B, -1) # [B, 384]
            all_avg_features.append(avg)
            all_max_features.append(max_feat)
        
        # 拼接所有层的平均池化特征用于层注意力 [B, 384 * 3]
        concatenated_avg_features = torch.cat(all_avg_features, dim=1) 
        
        # 计算层注意力权重 [B, 3]
        layer_weights = self.layer_attention(concatenated_avg_features)
        
        # 应用层注意力权重  [B, 384]
        fused_avg_features = sum(avg * weight.unsqueeze(1) for avg, weight in zip(all_avg_features, layer_weights.transpose(0, 1)))
        fused_max_features = sum(max_feat * weight.unsqueeze(1) for max_feat, weight in zip(all_max_features, layer_weights.transpose(0, 1)))
        
        # 应用特征注意力机制 [B, 384]
        attention_weights = self.feature_attention(fused_avg_features)
        attended_avg_features = fused_avg_features * attention_weights
        attended_max_features = fused_max_features * attention_weights
        
        # 组合不同类型的特征  [B, 384 * 2]
        classification_features = torch.cat([attended_avg_features, attended_max_features], dim=1)  # 组合注意力加权的GAP和GMP特征
        
        # 通过分类特征提取器
        classification_features = self.classification_backbone(classification_features)
        
        # 任务特定的分类头
        benign_malignant = self.benign_malignant_head(classification_features)
        tirads = self.tirads_head(classification_features)
        
        return out, benign_malignant, tirads


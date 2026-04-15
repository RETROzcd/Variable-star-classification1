import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.mha(x)
        x1 = self.norm1(x + attn_output)
        ffn_output = self.ffn(x1)
        x2 = self.norm2(x1 + ffn_output)
        return x2


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.input_proj = nn.Linear(input_dim, d_model)

    def forward(self, src, src_mask=None):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.mean(dim=1)


"""
class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, config['model']['d_model']) if config['model'].get('project_img', False) else None

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.proj(x) if self.proj else x
"""


class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.proj = nn.Linear(512, config['model']['d_model'])

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


"""class MultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 图像编码器
        self.img_encoder = CNNEncoder(config)

        # 非图像特征编码（period + num_peaks）
        self.non_img_encoder = nn.Sequential(
            nn.Linear(2, config['model']['d_model']),
            nn.GELU(),
            nn.LayerNorm(config['model']['d_model'])
        )

        # 时序编码器
        self.temporal_encoder = TemporalTransformer(
            input_dim=2,
            d_model=config['model']['temporal_dim'],
            nhead=config.get('temporal_nhead', 4),
            num_layers=config.get('temporal_layers', 3)
        )


        # 跨模态融合
        self.fusion = nn.Sequential(
            nn.Linear(config['model']['d_model'] * 2 + config['model']['temporal_dim'], 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        # 分类头
        self.classifier = nn.Linear(512, config['model']['num_classes'])

    def forward(self, img, time_series, non_img, masks):
        # 图像特征 [B, D]
        img_feat = self.img_encoder(img)

        # 非图像特征 [B, D]
        non_img_feat = self.non_img_encoder(non_img)

        # 时序特征 [B, T]
        ts_feat = self.temporal_encoder(time_series, src_mask=masks)

        # 特征拼接
        fused = torch.cat([img_feat, non_img_feat, ts_feat], dim=1)

        # 融合与分类
        fused = self.fusion(fused)
        return self.classifier(fused)
"""


class MultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 确保所有编码器输出维度一致
        self.img_encoder = CNNEncoder(config)  # 输出维度: d_model
        self.non_img_encoder = nn.Sequential(
            nn.Linear(2, config['model']['d_model']),  # 输出维度: d_model
            nn.GELU(),
            nn.LayerNorm(config['model']['d_model'])
        )

        # 统一时序编码器维度
        self.temporal_encoder = TemporalTransformer(
            input_dim=2,
            d_model=config['model']['d_model'],  # 使用统一维度
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers']
        )

        # 修正交叉注意力参数
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config['model']['d_model'],  # 与编码器维度一致
            num_heads=config['model']['nhead'],
            batch_first=True
        )

        # 修正融合层输入维度
        total_dim = config['model']['d_model'] * 3  # img + non_img + temporal
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),  # 输入维度需等于拼接后的总维度
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 分类器保持简单结构
        self.classifier = nn.Linear(512, config['model']['num_classes'])

    def forward(self, img, time_series, non_img, masks):
        # 获取各模态特征
        img_feat = self.img_encoder(img)  # (B, d_model)
        non_img_feat = self.non_img_encoder(non_img)  # (B, d_model)
        ts_feat = self.temporal_encoder(time_series)  # (B, d_model)

        # 交叉注意力（需统一序列长度）
        cross_feat, _ = self.cross_attn(
            query=img_feat.unsqueeze(1),  # (B, 1, d_model)
            key=ts_feat.unsqueeze(1),  # (B, 1, d_model)
            value=ts_feat.unsqueeze(1)  # (B, 1, d_model)
        )
        cross_feat = cross_feat.squeeze(1)  # (B, d_model)

        # 特征融合（维度验证）
        fused = torch.cat([img_feat, non_img_feat, ts_feat], dim=1)  # (B, 3*d_model)
        fused = self.fusion(fused)  # (B, 512)
        return self.classifier(fused)  # (B, num_classes)
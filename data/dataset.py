import os
import numpy as np
import ast
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import random
from sklearn.preprocessing import StandardScaler
from data.preprocessing import TimeSeriesProcessor
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
import torchvision.transforms as transforms


script_dir = os.path.dirname(os.path.abspath(__file__))

class StellarDataset(Dataset):
    def __init__(self, config, augment=False):
        self.image_dir = os.path.join(script_dir, config['data']['img_dir'])
        self.label_file = os.path.join(script_dir, config['data']['label_file'])
        self.batch_size = config['data']['batch_size']
        self.target_size = tuple(config['data']['img_size'])
        self.augment = augment
        self.ts_length = config['preprocess']['ts_length']
        self.smoothing_method = config['preprocess']['smoothing_method']

        # 加载数据
        self.data = pd.read_csv(self.label_file)
        self.oids = self.data['#oid'].values
        self.periods = self.data['period'].values.astype(float)
        self.num_peaks = self.data['num_peaks'].values.astype(int)
        self.classes = self.data['classALeRCE'].values

        # 解析时间序列
        """
        self.time_flux_pairs = []
        for row in self.data:
            time = np.fromstring(row[3], sep=',', dtype=np.float32)
            flux = np.fromstring(row[4], sep=',', dtype=np.float32)
            self.time_flux_pairs.append((time, flux))
        self.time = self.data['time'].apply(lambda x: np.array(list(map(float, x.split(',')))))
        self.flux = self.data['flux'].apply(lambda x: np.array(list(map(float, x.split(',')))))
        """
        self.time = self.data['time'].apply(lambda x: np.array(x.strip('[]').split(','), dtype=float))
        self.flux = self.data['flux'].apply(lambda x: np.array(x.strip('[]').split(','), dtype=float))


        # 特征标准化
        self.scaler = StandardScaler()
        self.non_img_features = self.scaler.fit_transform(
            np.column_stack([self.periods, self.num_peaks])
        )

        # 标签编码
        self.unique_classes = np.unique(self.classes)  # 获取所有唯一类别
        self.class_to_idx = {cls: i for i, cls in enumerate(self.unique_classes)}
        self.labels = np.array([self.class_to_idx[cls] for cls in self.classes])

        # 计算类别权重
        class_counts = {cls: np.sum(self.labels == self.class_to_idx[cls]) for cls in self.unique_classes}
        self.class_weights = torch.FloatTensor([1.0 / (class_counts[cls] + 1e-8) for cls in self.unique_classes])  # 避免除以零

        # 初始化时间序列处理器
        self.ts_processor = TimeSeriesProcessor(
            target_length=self.ts_length,
            smoothing_method=self.smoothing_method
        )

    def __len__(self):
        return len(self.oids)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, f"{self.oids[idx]}.png")
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            img = Image.new('RGB', self.target_size)

        # 数据增强
        if self.augment:
            img = self._augment_image(img)

        # 图像预处理
        img = img.resize(self.target_size)
        img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # 非图像特征
        non_img_tensor = torch.tensor(self.non_img_features[idx], dtype=torch.float32)

        # 时间序列处理
        time = self.time[idx]
        flux = self.flux[idx]
        ts_feature = self.ts_processor.process_sequence(time, flux)

        # 标签
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'image': img_tensor,
            'features': non_img_tensor,
            'time_series': torch.FloatTensor(ts_feature),
            'label': label_tensor
        }


    def _augment_image(self, img):
        """增强策略"""
        # 随机翻转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 颜色扰动
        enhancers = [
            (ImageEnhance.Brightness, 0.8, 1.2),
            (ImageEnhance.Contrast, 0.8, 1.2),
            (ImageEnhance.Sharpness, 0.8, 1.2)
        ]
        for enhancer, min_val, max_val in enhancers:
            if random.random() < 0.3:
                factor = random.uniform(min_val, max_val)
                img = enhancer(img).enhance(factor)

        # 随机旋转
        if random.random() < 0.3:
            angle = random.choice([-30, -15, 15, 30])
            img = img.rotate(angle)

        if random.random() < 0.3:
            w, h = img.size
            new_w = int(w * random.uniform(0.8, 1.0))
            new_h = int(h * random.uniform(0.8, 1.0))
            img = transforms.RandomCrop((new_h, new_w))(img)
            img = img.resize((w, h))  # 恢复原始尺寸

        return img

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    @staticmethod
    def collate_fn(batch):
        """改进的批次处理"""
        images = torch.stack([x['image'] for x in batch])
        features = torch.stack([x['features'] for x in batch])
        labels = torch.stack([x['label'] for x in batch])
        # 处理时序数据
        ts_data = [x['time_series'] for x in batch]
        ts_padded, ts_mask = TimeSeriesProcessor().batch_process(
            [(t[:, 0], t[:, 1]) for t in ts_data]
        )
        return {
            'images': images,
            'features': features,
            'time_series': ts_padded,
            'masks': ts_mask,
            'labels': labels
        }
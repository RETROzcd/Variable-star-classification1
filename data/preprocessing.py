import torch
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class TimeSeriesProcessor:
    def __init__(self, target_length=512, smoothing_window=51, smoothing_method='savgol'):
        self.target_length = target_length  # 目标序列长度
        self.smoothing_window = smoothing_window  # 平滑窗口大小
        self.smoothing_method = smoothing_method  # 平滑方法
        self.scaler = StandardScaler()

    def _denoise(self, series):
        """降噪处理"""
        # 使用选定的平滑方法
        if self.smoothing_method == 'savgol':
            try:
                # Savitzky-Golay滤波器
                return savgol_filter(series, window_length=self.smoothing_window, polyorder=3)
            except ValueError:
                # 如果窗口大小超过数据长度，直接返回原数据
                return series
        elif self.smoothing_method == 'median':
            return medfilt(series, kernel_size=3)
        else:
            return series

    def _normalize(self, series):
        """归一化处理"""
        return self.scaler.fit_transform(series.reshape(-1, 1)).flatten()

    def _resample(self, series, time):
        """重采样处理"""
        if len(series) > self.target_length:
            # 裁剪
            start = np.random.randint(0, len(series) - self.target_length)
            series = series[start:start + self.target_length]
            time = time[start:start + self.target_length]
        elif len(series) < self.target_length:
            # 填充
            pad_width = self.target_length - len(series)
            series = np.pad(series, (0, pad_width), mode='edge')
            time = np.pad(time, (0, pad_width), mode='edge')
        return series, time

    def process_sequence(self, time, flux):
        """处理时间序列数据"""
        # 检查数据长度
        if len(flux) < 2 or len(time) < 2:
            return np.zeros((self.target_length, 2))

        # 降噪
        flux = self._denoise(flux)

        # 归一化
        flux = self._normalize(flux)

        # 重采样
        flux, time = self._resample(flux, time)

        # 时间归一化
        time = (time - time.min()) / (time.max() - time.min())

        return np.column_stack((time, flux))

    def batch_process(self, batch):
        """批量处理时间序列数据"""
        processed_sequences = []
        masks = []

        for seq in batch:
            # 获取序列
            series, time = seq
            processed_seq = self.process_sequence(time, series)
            processed_sequences.append(processed_seq)

            # 计算掩码
            mask = np.ones(self.target_length, dtype=bool)
            masks.append(mask)

        # 转换为 PyTorch 张量
        padded_sequences = pad_sequence(
            [torch.tensor(seq).float() for seq in processed_sequences],
            batch_first=True,
            padding_value=0.0
        )

        masks = torch.tensor(np.array(masks), dtype=torch.bool)

        return padded_sequences, masks
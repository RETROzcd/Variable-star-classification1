import torch
import yaml
import os
from models.transformer import MultiModalModel
from torch.utils.data import DataLoader, random_split
from data.dataset import StellarDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, recall_score
from utils.logger import Logger
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report
from utils.visualize import Visualize
import torch.cuda.amp
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

cudnn.benchmark = True
cudnn.deterministic = False

script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'configs', 'default.yaml')


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss.mean()


class Trainer:
    def __init__(self, config_path=config_file):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # 初始化日志
        self.logger = Logger(log_dir=os.path.join(script_dir, self.config['experiment']['log_dir']))
        self.logger.info("Logger initialized.")  # 打印日志信息

        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")  # 打印设备信息

        # 初始化scaler
        self.scaler = GradScaler()

        # 加载数据
        self.dataset = StellarDataset(self.config, augment=True)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = random_split(self.dataset, [train_size, val_size])
        self.logger.info(
            f"Dataset loaded. Train set size: {len(self.train_set)}, Val set size: {len(self.val_set)}")  # 打印数据集信息

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=self.dataset.collate_fn,  # 添加collate_fn
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=self.dataset.collate_fn  # 添加collate_fn
        )

        # 初始化损失函数
        """
        if self.dataset.class_weights is not None:
            self.class_weights = self.dataset.class_weights.to(self.device)
        else:
            self.class_weights = None
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        """
        self.criterion = FocalLoss(
            alpha=torch.tensor([0.2, 0.8], device=self.device),  # 根据类别分布调整
            gamma=2
        )

        # 初始化模型
        self.model = MultiModalModel(self.config).to(self.device)
        self.logger.info("Model initialized.")  # 打印模型信息

        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['lr']),
            weight_decay=float(self.config['training']['weight_decay'])
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )
        """
        # 在初始化调度器时使用已创建的loader
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=float(self.config['training']['lr']),
            epochs=self.config['training']['epochs'],
            steps_per_epoch=len(self.train_loader),  # 使用实例变量
            pct_start=0.3
        )

        self.criterion = torch.nn.CrossEntropyLoss(
            weight=self.dataset.class_weights.to(self.device)
        )
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=self.dataset.class_weights.to(self.device) if self.dataset.class_weights is not None else None,
            # label_smoothing=0.1  # 添加标签平滑
        )"""

        self.scaler = GradScaler(enabled=self.config['training']['mixed_precision'])
        self.logger.info("Optimizer, scheduler, criterion and scaler initialized.")  # 打印优化器等信息

    def train_epoch(self, loader):
        self.model.train()
        torch.cuda.empty_cache()
        total_loss, correct = 0, 0
        all_preds, all_labels = [], []
        gradient_accumulation_steps = 8

        for step, batch in enumerate(loader):
            img = batch['images'].to(self.device)
            non_img = batch['features'].to(self.device)
            time_series = batch['time_series'].to(self.device)
            masks = batch['masks'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 启用混合精度
            with autocast():
                outputs = self.model(img, time_series, non_img, masks)
                loss = self.criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(loader) - 1:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        metrics = {
            'loss': total_loss / len(loader),
            'acc': correct / len(loader.dataset),
            'f1': f1_score(all_labels, all_preds, average='macro')
        }
        return metrics

    def validate(self, loader):
        self.model.eval()
        torch.cuda.empty_cache()
        total_loss, correct = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                img = batch['images'].to(self.device)
                non_img = batch['features'].to(self.device)
                time_series = batch['time_series'].to(self.device)
                masks = batch['masks'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(img, time_series, non_img, masks)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.dataset.unique_classes,
            output_dict=True
        )

        metrics = {
            'loss': total_loss / len(loader),
            'acc': correct / len(loader.dataset),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'report': report
        }
        return metrics

    # 在Trainer类中添加学习率监控方法
    def log_learning_rate(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Current learning rate: {current_lr:.2e}")

    def run(self):
        best_f1 = 0

        patience = self.config.get('training', {}).get('early_stopping_patience', 5)  # 增加耐心值
        counter = 0  # 计数器，记录验证集指标没有提升的轮数

        # 用于记录每个 epoch 的训练和验证指标
        train_losses = []
        train_accs = []
        train_f1s = []
        val_losses = []
        val_accs = []
        val_f1s = []
        learning_rates = []

        for epoch in range(self.config['training']['epochs']):
            train_metrics = self.train_epoch(self.train_loader)
            val_metrics = self.validate(self.val_loader)

            # 更新学习率
            self.scheduler.step(val_metrics['f1'])

            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)  # 将学习率添加到列表

            # 记录每个 epoch 的指标
            train_losses.append(train_metrics['loss'])
            train_accs.append(train_metrics['acc'])
            train_f1s.append(train_metrics['f1'])
            val_losses.append(val_metrics['loss'])
            val_accs.append(val_metrics['acc'])
            val_f1s.append(val_metrics['f1'])

            # 日志记录
            log = f"Epoch {epoch + 1}/{self.config['training']['epochs']}\n" + \
                  f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f}\n" + \
                  f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f}"

            self.logger.info(log)

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.model.state_dict(),
                           os.path.join('models', f'{self.config["experiment"]["name"]}_best.pth'))
                self.logger.info(f"Best model saved at epoch {epoch + 1} with F1 score: {best_f1:.4f}")
                counter = 0  # 重置计数器
            else:
                counter += 1

            # 早停机制
            if counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

            """
            # 早停机制：仅在达到准确率阈值后触发
            if has_reached_threshold and counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement since reaching acc 0.8.")
                break
            """

        self.logger.info(f"Training completed. Best Val F1: {best_f1:.4f}")

        # 创建一个简单的历史记录字典，用于可视化
        history = {
            'loss': train_losses,
            'val_loss': val_losses,
            'accuracy': train_accs,
            'val_accuracy': val_accs,
            'learning_rate': learning_rates
        }

        # 调用可视化工具
        visualizer = Visualize()
        visualizer.plot_training_history(history)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
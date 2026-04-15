import matplotlib.pyplot as plt

class Visualize:
    @staticmethod
    def plot_training_history(history):
        """
        绘制训练损失、准确率和学习率曲线。
        """
        plt.figure(figsize=(18, 6))

        # 绘制损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(history['loss'], label='Training Loss')  # 修改为直接使用 history
        plt.plot(history['val_loss'], label='Validation Loss')  # 修改为直接使用 history
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')  # 修改为直接使用 history
        plt.plot(history['val_accuracy'], label='Validation Accuracy')  # 修改为直接使用 history
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # 绘制学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(history['learning_rate'], label='Learning Rate')  # 修改为直接使用 history
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.show()
        plt.close()
import logging
from datetime import datetime
import os

class Logger:
    def __init__(self, log_dir='./logs/'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'train_log_{current_time}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

    def info(self, message):
        logging.info(message)

    def warning(self, message):
        logging.warning(message)

    def error(self, message):
        logging.error(message)

    def save(self):
        logging.shutdown()
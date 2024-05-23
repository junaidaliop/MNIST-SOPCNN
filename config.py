# config.py

import os
import torch

class Config:
    def __init__(self):
        self.batch_size = 256
        self.learning_rate = 0.001
        self.epochs = 2000
        self.optimizer = 'adam'
        self.model_name = 'SOPCNN'
        self.num_classes = 10
        self.data_path = './data'
        self.results_path = './results'
        self.model_save_path = os.path.join(self.results_path, 'MNIST_SOPCNN.pth')
        self.log_interval = 10
        self.patience = 10
        self.min_delta = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
os.makedirs(config.results_path, exist_ok=True)

# main.py

import torch
from config import config
from data.dataset import get_train_loader, get_test_loader
from models.model import SOPCNN
from train import train
from test import test

def main():
    # Load data
    train_loader = get_train_loader(config.batch_size, config.data_path)
    val_loader = get_test_loader(config.batch_size, config.data_path)  # Use the same test loader for validation in this example
    
    # Initialize model, optimizer
    model = SOPCNN(num_classes=config.num_classes).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Train and Test
    train(model, train_loader, val_loader, optimizer, config)
    test(model, val_loader, config)

if __name__ == '__main__':
    main()

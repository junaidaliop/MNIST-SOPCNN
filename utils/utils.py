# utils/utils.py

import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    val_accuracy = checkpoint['val_accuracy']
    return model, optimizer, epoch, val_loss, val_accuracy

def log_metrics(log_file, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
    with open(log_file, 'a') as f:
        f.write(f'{epoch},{train_loss},{train_accuracy},{val_loss},{val_accuracy}\n')
    print(f'Logged metrics to {log_file}: epoch={epoch}, train_loss={train_loss}, train_accuracy={train_accuracy}, val_loss={val_loss}, val_accuracy={val_accuracy}')

def plot_metrics(log_df, save_path):
    print(f'Log DataFrame columns: {log_df.columns}')
    print(f'Log DataFrame head: \n{log_df.head()}')

    plt.figure(figsize=(10, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(log_df['epoch'], log_df['train_loss'], label='Training Loss')
    plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(log_df['epoch'], log_df['train_accuracy'], label='Training Accuracy')
    plt.plot(log_df['epoch'], log_df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_validation_metrics.png'))
    plt.show()
# utils/early_stopping.py

import torch

class EarlyStopping:
    def __init__(self, baseline=95, patience=5, min_delta=0.01):
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_accuracy = None
        self.early_stop = False

    def __call__(self, val_loss, val_accuracy, model, optimizer, epoch, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_accuracy = val_accuracy
            self.save_checkpoint(val_loss, val_accuracy, model, optimizer, epoch, path)
        elif val_accuracy >= self.baseline:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.best_accuracy = val_accuracy
                self.save_checkpoint(val_loss, val_accuracy, model, optimizer, epoch, path)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.save_checkpoint(val_loss, val_accuracy, model, optimizer, epoch, path)

    def save_checkpoint(self, val_loss, val_accuracy, model, optimizer, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, path)

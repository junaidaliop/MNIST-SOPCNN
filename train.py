# train.py

import os
import torch
import torch.nn.functional as F
import pandas as pd
from utils.utils import save_checkpoint, log_metrics, plot_metrics
from utils.early_stopping import EarlyStopping

def train(model, train_loader, val_loader, optimizer, config):
    early_stopping = EarlyStopping(baseline=99.85, patience=config.patience, min_delta=config.min_delta)
    training_log = []
    
    model.to(config.device)
    
    # Create log file with headers
    log_file = os.path.join(config.results_path, 'training_log.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_accuracy,val_loss,val_accuracy\n')
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % config.log_interval == 0:
                train_accuracy = 100. * correct / ((batch_idx + 1) * len(data))
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {train_accuracy:.2f}%')
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct / len(train_loader.dataset)
        
        val_loss, val_accuracy = validate(model, val_loader, config.device)
        
        print(f'\nEpoch: {epoch}\tTraining Loss: {train_loss:.6f}\tTraining Accuracy: {train_accuracy:.2f}%\t'
              f'Validation Loss: {val_loss:.6f}\tValidation Accuracy: {val_accuracy:.2f}%\n')
        
        training_log.append((epoch, train_loss, train_accuracy, val_loss, val_accuracy))
        
        # Log metrics
        log_metrics(log_file, epoch, train_loss, train_accuracy, val_loss, val_accuracy)
        
        # Check early stopping criteria
        early_stopping(val_loss, val_accuracy, model, optimizer, epoch, config.model_save_path)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Plot metrics
    training_log_df = pd.read_csv(log_file)
    plot_metrics(training_log_df, config.results_path)

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, val_accuracy
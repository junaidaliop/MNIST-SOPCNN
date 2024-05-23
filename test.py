# test.py

import os
import torch
import torch.nn.functional as F
import pandas as pd

def test(model, test_loader, config):
    model.eval()
    test_loss = 0
    correct = 0
    test_log = []
    device = config.device
    
    model.to(device)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum').item()
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_log.append((loss, correct))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.0f}%\n')
    
    # Save test log
    test_log_df = pd.DataFrame(test_log, columns=['loss', 'correct'])
    test_log_df.to_csv(os.path.join(config.results_path, 'test_log.csv'), index=False)

    return test_loss, test_accuracy

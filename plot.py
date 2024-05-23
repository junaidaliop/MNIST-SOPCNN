import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_metrics(log_file, save_path):
    log_df = pd.read_csv(log_file)
    os.makedirs(save_path, exist_ok=True)

    rcParams.update({
        'font.size': 14,
        'figure.figsize': (6, 4),
        'figure.dpi': 600,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.titleweight': 'bold',
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'text.usetex': True,
        'axes.grid': True,
        'grid.color': 'gray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5
    })

    plt.figure()
    plt.plot(log_df['epoch'], log_df['train_loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.xlabel(r'\textit{Epoch}')
    plt.ylabel(r'\textit{Loss}')
    plt.title('MNIST-SOPCNN')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'), dpi=600)
    plt.close()

    best_val_acc = log_df['val_accuracy'].max()
    best_val_epoch = log_df['val_accuracy'].idxmax()

    plt.figure()
    plt.plot(log_df['epoch'], log_df['train_accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(log_df['epoch'], log_df['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    plt.scatter(best_val_epoch, best_val_acc, color='green', s=50, zorder=5)
    plt.annotate(
        f'Best: {best_val_acc:.2f}\\%',
        xy=(best_val_epoch, best_val_acc),
        xytext=(best_val_epoch, best_val_acc - 2),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        fontsize=12,
        color='green',
        ha='center',
        fontweight='bold'
    )
    plt.xlabel(r'\textit{Epoch}')
    plt.ylabel(r'\textit{Accuracy}')
    plt.title('MNIST-SOPCNN')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'), dpi=600)
    plt.close()

if __name__ == "__main__":
    log_file = 'results/training_log.csv'
    save_path = 'results/plots'
    plot_metrics(log_file, save_path)

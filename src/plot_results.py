import torch
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_history(history, save_path):
    """Plots and saves training and validation metrics."""
    # Convert any potential GPU tensors in history to CPU
    cpu_history = {k: [item.cpu().item() if torch.is_tensor(item) else item for item in v] for k, v in history.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(cpu_history['train_loss'], label='Train Loss')
    ax1.plot(cpu_history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(cpu_history['train_acc'], label='Train Acc')
    ax2.plot(cpu_history['val_acc'], label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot training history from a saved model checkpoint.")
    parser.add_argument("model_path", type=str, help="Path to the saved .pth model checkpoint file.")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))

    if 'history' in checkpoint:
        history = checkpoint['history']
        model_name = os.path.basename(args.model_path).replace('.pth', '')
        save_path = f"reports/{model_name}_history.png"
        plot_training_history(history, save_path)
    else:
        print("Error: No 'history' dictionary found in the saved model file.")

if __name__ == '__main__':
    main()
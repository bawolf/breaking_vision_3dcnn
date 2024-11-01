import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
from dataset import SimpleVideoDataset
from model import Simple3DCNN

def load_run_config(run_dir):
    """Load run configuration from JSON"""
    config_path = os.path.join(run_dir, 'run_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def plot_training_curves(log_file, output_dir):
    data = pd.read_csv(log_file)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['train_accuracy'], label='Train Accuracy')
    plt.plot(data['epoch'], data['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def generate_confusion_matrix(model, data_loader, device, output_dir):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in data_loader:
            videos = videos.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    return cm

if __name__ == "__main__":
    # Find the most recent run directory
    base_dir = 'outputs'
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not run_dirs:
        raise FileNotFoundError("No run directories found")
    
    latest_run = max(run_dirs)
    run_dir = os.path.join(base_dir, latest_run)
    print(f"Using run directory: {run_dir}")

    # Load configuration
    config = load_run_config(run_dir)
    print("Loaded configuration:")
    print(json.dumps(config, indent=2))

    # Paths
    log_file = os.path.join(run_dir, 'training_log.csv')
    model_path = os.path.join(run_dir, 'best_model.pth')
    test_csv = "test.csv"  # You might want to add this to config as well
    
    # Create a directory for visualization outputs
    vis_dir = os.path.join(run_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot training curves
    plot_training_curves(log_file, vis_dir)
    
    # Load model
    model = Simple3DCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create test dataset and dataloader using config
    test_dataset = SimpleVideoDataset(
        data_file=test_csv,
        root_dir=config['dataset_root']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False
    )
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(model, test_loader, device, vis_dir)

        # Calculate and print per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for class_idx, accuracy in enumerate(per_class_accuracy):
        print(f"Class {class_idx} accuracy: {accuracy:.2%}")
    print(f"Overall accuracy: {cm.diagonal().sum() / cm.sum():.2%}")
    
    print(f"Visualization complete! Check the output directory: {vis_dir}")
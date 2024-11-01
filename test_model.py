import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SimpleVideoDataset
from model import Simple3DCNN
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = total_loss / len(test_loader)
    return accuracy, average_loss

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    base_dir = 'outputs'
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    latest_run = max(run_dirs)
    run_dir = os.path.join(base_dir, latest_run)
    # Load the trained model
    model_path = os.path.join(run_dir, 'best_model.pth')
    model = Simple3DCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    logger.info(f"Loaded model from {model_path}")

    # Create test dataset and dataloader
    test_csv = "test.csv"  # Path to your test CSV file
    test_data_root = "../finetune/inputs"  # Path to your test data root directory
    test_dataset = SimpleVideoDataset(test_csv, test_data_root)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Test the model
    accuracy, average_loss = test_model(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Test Loss: {average_loss:.4f}")

    # Calculate per-class accuracy
    class_correct = list(0. for i in range(3))  # Assuming 3 classes
    class_total = list(0. for i in range(3))
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i, label in enumerate(labels):
                class_correct[label.item()] += c[i].item()
                class_total[label.item()] += 1

    for i in range(3):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            logger.info(f'Accuracy of class {i}: {class_accuracy:.2f}%')
        else:
            logger.info(f'No samples for class {i}')

if __name__ == "__main__":
    main()
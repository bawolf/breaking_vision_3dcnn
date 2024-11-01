import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SimpleVideoDataset
from early_stopping import EarlyStopping
from model import Simple3DCNN
import logging
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch import load as torch_load
from utils import create_run_directory
import os
import json
from datetime import datetime

# Create a directory for this run
run_dir = create_run_directory()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(os.path.join(run_dir, 'training.log')),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 100
patience = 10  # for early stopping

#Dataset root directory
dataset_root = "../finetune/3moves_balanced"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create datasets
train_dataset = SimpleVideoDataset("train.csv", dataset_root)
val_dataset = SimpleVideoDataset("val.csv", dataset_root)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model, loss function, and optimizer
model = Simple3DCNN().to(device)
logger.info(f"Model architecture:\n{model}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Initialize early stopping
early_stopping = EarlyStopping(patience=patience)

# Open a CSV file to log training progress
csv_path = os.path.join(run_dir, 'training_log.csv')
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

best_val_loss = float('inf')
best_model_path = os.path.join(run_dir, 'best_model.pth')

# Save run information
def save_run_info(run_dir, config):
    """Save run configuration as JSON"""
    config_path = os.path.join(run_dir, 'run_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Create config dictionary
config = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "num_epochs": num_epochs,
    "patience": patience,
    "device": str(device),
    "dataset_root": dataset_root,
}

# Save run info
save_run_info(run_dir, config)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_accuracy = 0.0
    total_batches = len(train_loader)

    # Use tqdm for a progress bar
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (videos, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}")

            videos = videos.to(device)
            labels = labels.to(device)

            # Step 1: Forward pass
            outputs = model(videos)

            # Step 2: Compute the loss
            loss = criterion(outputs, labels)

            # Step 3: Backward pass (backpropagation)
            optimizer.zero_grad()  # Zero the gradient buffers
            loss.backward()  # This is where backpropagation happens!

            # Step 4: Update the weights
            optimizer.step()

            # Calculate batch accuracy
            accuracy = calculate_accuracy(outputs, labels)

            # Update running loss and accuracy
            running_loss += loss.item()
            running_accuracy += accuracy

            # Update progress bar
            tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy)

    # Calculate epoch loss and accuracy
    train_loss = running_loss / total_batches
    train_accuracy = 100. * running_accuracy / total_batches

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_total_batches = len(val_loader)

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            accuracy = calculate_accuracy(outputs, labels)

            val_loss += loss.item()
            val_accuracy += accuracy

    val_loss /= val_total_batches
    val_accuracy = 100. * val_accuracy / val_total_batches

    # Log the metrics
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Write to CSV
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, train_accuracy, val_loss, val_accuracy])

    # Learning rate scheduling
    scheduler.step(val_loss)

    if patience > 0:
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=True)
        logger.info(f"Saved best model to {best_model_path}")

    # Overfitting detection
    if train_accuracy - val_accuracy > 10:  # If training accuracy is 10% higher than validation
        logger.warning("Possible overfitting detected")

logger.info("Training finished!")

# Save the final model
final_model_path = os.path.join(run_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path, _use_new_zipfile_serialization=True)
logger.info(f"Saved final model to {final_model_path}")
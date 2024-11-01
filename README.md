# Simple 3D CNN for Action Recognition

## Setup

### Prerequisites

- Python 3.11
- NVIDIA GPU with CUDA support
- Miniconda or Anaconda

### Environment Setup

1. Create and activate a new conda environment:
```bash
conda create -n learning python=3.11
conda activate learning
```

2. Install PyTorch and related packages:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```
# Project Structure

- `dataset.py`: Implements the SimpleVideoDataset class for video data handling
- `model.py`: Contains the Simple3DCNN architecture
- `training.py`: Training loop implementation
- `test_model.py`: Model evaluation script
- `visualize.py`: Visualization script to measure model performance and generate confusion matrix
- `utils.py`: Helper functions
- `outputs/`: Directory for storing model checkpoints and logs

## Usage

### Training


#### Hyperparameters
Update the training.py script to take in the correct hyperparameters

- `batch_size`: number of videos per batch
- `learning_rate`: learning rate for the optimizer
- `num_epochs`: number of epochs to train for
- `patience`: number of epochs to wait before early stopping (0 to disable)

Also update the dataset root directory in the training.py script
- `dataset_root`: root directory of the dataset

Dataset files:
- `train.csv`: training dataset
- `val.csv`: validation dataset
- `test.csv`: test dataset

Format of the dataset files(csv):
```
<video_path>, <label>
<video_path>, <label>
<video_path>, <label>
...
```

#### Training Script

```bash
python training.py
```

## License

[Add your license information here]

## Contact

[Add your contact information if you want]

#TODO
update script to take true csvs instead of weird txt format
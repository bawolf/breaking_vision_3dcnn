import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(16 * 8 * 112 * 112, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch_size, num_frames, channels, height, width]
        # Reshape to [batch_size, channels, num_frames, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
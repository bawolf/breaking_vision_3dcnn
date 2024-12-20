import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import logging
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVideoDataset(Dataset):
    def __init__(self, data_file, root_dir, num_frames=16):
        self.data = []
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with open(os.path.join(self.root_dir, data_file), 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) == 2:
                    path, label = row
                    full_video_path = self.get_video_path(path)

                    if os.path.exists(full_video_path):
                        self.data.append((full_video_path, int(label)))
                    else:
                        logger.warning(f"File not found: {full_video_path}")

    def __len__(self):
        return len(self.data)
    
    def get_video_path(self, relative_path):
        return os.path.join(self.root_dir, relative_path)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
                
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
            
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # If the video has fewer frames than we need, we'll loop the video
        if frame_count < self.num_frames:
            repeat = self.num_frames // frame_count + 1
        else:
            repeat = 1
        
        frame_indices = np.linspace(0, frame_count - 1, self.num_frames // repeat, dtype=int)
        
        for _ in range(repeat):
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.transform(frame)
                    frames.append(frame)
                if len(frames) == self.num_frames:
                    break
            if len(frames) == self.num_frames:
                break
        
        cap.release()
        
        # If we couldn't get enough frames, we'll duplicate the last frame
        if len(frames) == 0:
            raise ValueError(f"Failed to read any frames from video: {video_path}")
            
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        
        video = torch.stack(frames, dim=0)
        return video, label
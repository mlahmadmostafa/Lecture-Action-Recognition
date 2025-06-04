from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import cv2
import torch
import pandas as pd
import pickle
# --- Preprocessing Utility Function (Same as before) ---
def preprocess_image_for_yolo_backbone(img):
    """
    Loads an image, preprocesses it for a YOLO backbone (RGB, resized, normalized),
    and returns it as a PyTorch tensor.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired (width, height) for the image. YOLOv8 typically uses 640x640.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (3, H, W), normalized to [0, 1].
                      Returns None if image loading fails.
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img / 255.0

    return img


class ActionLoad(Dataset):
    def __init__(self, root_dir,df_path): # root_dir = "Dataset_2"
        self.root_dir = root_dir
        self.df_path = df_path
        self.df_class_map = pd.read_csv(f"{root_dir}/class_number_map.csv")
    def __len__(self):
        return len(self.df_path)
    def get_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        frames = []
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            return torch.empty(0)  # or raise an Exception
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = preprocess_image_for_yolo_backbone(frame)
            if not ret:
                break
            frames.append(frame)
            i += 1
        
        cap.release()   
        frames = torch.stack(frames, dim=0)  
        
        if len(frames) == 0:
            print(f"[Warning] No frames read from video: {video_path}")
            return torch.empty(0)
    
        return frames
    def __getitem__(self,idx):
        sample = self.df_path.iloc[idx]
        video_path = sample["new_video_path"]
        class_name = sample["class"]
        class_number = self.df_class_map[self.df_class_map["class"] == class_name]["number"].values[0]
        frames = self.get_video(video_path)
        return frames, torch.tensor([class_number], dtype=torch.long)
        
    
if __name__ == "__main__":
    dataloader = ActionLoad(root_dir="Dataset_2")
    for image, anno in dataloader:
        print(image.shape, anno)
        break
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
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img / 255.0

    return img


class ActionLoad(Dataset):
    def __init__(self, root_dir, samples_list_of_lists):
        self.root_dir = root_dir
        self.samples_map = samples_list_of_lists # list of lists [['around01_1.jpg', ...], ['around01_2.jpg', ...] ...]
    def __len__(self):
        return len(self.samples_map)
    def get_image(self, image_name):
        image_name = image_name.split('.')[0]
        image = Image.open(f"{self.root_dir}/PPDataset/train/images/{image_name}.jpg")
        image = preprocess_image_for_yolo_backbone(image)
        with open(f"{self.root_dir}/PPDataset/train/labels/{image_name}.txt", 'r') as file:
            anno = torch.tensor(int(file.read().split(' ')[0]), dtype=torch.long)        
        return image, anno
    
    def __getitem__(self,idx):
        images = []
        annotations = []
        for image_name in self.samples_map[idx]:
            image, anno = self.get_image(image_name)
            images.append(image)
            annotations.append(anno)
        annotations = torch.stack(annotations, dim=0)
        images = torch.stack(images, dim=0)
        counts = torch.bincount(annotations)
        most_frequent_annotation = torch.argmax(counts)
        
        return images, most_frequent_annotation
        
    
if __name__ == "__main__":
    os.chdir("..")
    image_list = os.listdir("Dataset/PPDataset/train/images")
    image_list = [image.split('.')[0] for image in image_list]
    dataloader = ActionLoad(root_dir="Dataset", images_list=image_list)
    for image, anno in dataloader:
        print(image.shape, anno)
        break
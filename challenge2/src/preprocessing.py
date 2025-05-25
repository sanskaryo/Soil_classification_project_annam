"""
Author: sanskar khandelwal
Team Name: TheLastTransformer
Team Members: 1
Leaderboard Rank: 56
"""


# preprocessing.py
# Handles:

# Dataset loading

# Image preprocessing

# Feature extraction using ResNet18

# Saving extracted features



import os
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

#  Load pretrained ResNet18 (remove classifier head)
def load_resnet(device):
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()
    return resnet

#  Image transformations
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

#  Custom dataset class
class SoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image_id']
        image_path = os.path.join(self.img_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_id

#  Feature extractor
def extract_features(dataloader, model, device):
    features = []
    ids = []
    with torch.no_grad():
        for images, image_ids in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images).cpu().numpy()
            features.append(feats)
            ids.extend(image_ids)
    return np.vstack(features), ids

#  Entrypoint
if __name__ == "__main__":
    # Paths
    train_csv = '/kaggle/input/soil-classification-part-2/soil_competition-2025/train_labels.csv'
    test_csv = '/kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv'
    train_dir = '/kaggle/input/soil-classification-part-2/soil_competition-2025/train'
    test_dir = '/kaggle/input/soil-classification-part-2/soil_competition-2025/test'

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CSVs and transforms
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    transform = get_transform()

    # Load model
    resnet = load_resnet(device)

    # Create datasets and loaders
    train_loader = DataLoader(SoilDataset(train_df, train_dir, transform), batch_size=32, shuffle=False)
    test_loader = DataLoader(SoilDataset(test_df, test_dir, transform), batch_size=32, shuffle=False)

    # Extract features
    train_features, _ = extract_features(train_loader, resnet, device)
    test_features, test_ids = extract_features(test_loader, resnet, device)

    # Save extracted features
    np.save("train_features.npy", train_features)
    np.save("test_features.npy", test_features)
    np.save("test_ids.npy", test_ids)
    print(" Saved train/test features and test image IDs.")


# .npy files are in git ignore so wont be visisble in the repo in github
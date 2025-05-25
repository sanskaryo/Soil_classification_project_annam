"""
Author: sanskar khandelwal
Team Name: TheLastTransformer
Team Members: 1
Leaderboard Rank: 56
"""


# preprocessing.py

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# 📂 Paths
data_root = "/kaggle/input/soil-classification/soil_classification-2025"
train_img_dir = os.path.join(data_root, "train")
train_csv = os.path.join(data_root, "train_labels.csv")

# 📄 Load train CSV
df_train = pd.read_csv(train_csv)

# 🔤 Label encoding
label_encoder = LabelEncoder()
df_train['label'] = label_encoder.fit_transform(df_train['soil_type'])
class_names = label_encoder.classes_
np.save("label_encoder_classes.npy", class_names)

# ⚙️ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 Transformations
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🧠 ResNet18 feature extractor
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = torch.nn.Identity()
resnet18 = resnet18.to(device)
resnet18.eval()

def extract_features(img_paths):
    features = []
    for path in tqdm(img_paths, desc="Extracting features"):
        img = Image.open(path).convert('RGB')
        img_tensor = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet18(img_tensor).cpu().numpy().flatten()
        features.append(feat)
    return np.array(features)

# 📷 Train image paths
train_paths = [os.path.join(train_img_dir, fname) for fname in df_train['image_id']]

# 🔍 Feature extraction
X = extract_features(train_paths)
y = df_train['label'].values

# 💾 Save features
np.save("X_train_features.npy", X)
np.save("y_train_labels.npy", y)
print("✅ Preprocessing complete. Features and labels saved.")

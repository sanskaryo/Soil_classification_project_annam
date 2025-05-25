"""
Author: sanskar khandelwal
Team Name: TheLastTransformer
Team Members: 1
Leaderboard Rank: 56
"""


# postprocessing.py

import os
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import joblib

# 📂 Paths
data_root = "/kaggle/input/soil-classification/soil_classification-2025"
test_img_dir = os.path.join(data_root, "test")
test_csv = os.path.join(data_root, "test_ids.csv")

# 📄 Load saved features and labels
X = np.load("X_train_features.npy")
y = np.load("y_train_labels.npy")
class_names = np.load("label_encoder_classes.npy")

# 🔁 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X[tr_idx], y[tr_idx])
    preds = clf.predict(X[val_idx])
    f1 = f1_score(y[val_idx], preds, average='macro')
    fold_scores.append(f1)
    print(f"\nFold {fold+1} F1-score: {f1:.4f}")
    print(classification_report(y[val_idx], preds, target_names=class_names))

# 📊 Plot fold scores
plt.figure(figsize=(8, 5))
plt.bar([f"Fold {i+1}" for i in range(5)], fold_scores, color='skyblue')
plt.ylim(0, 1)
plt.title("F1 Scores per Fold")
plt.ylabel("Macro F1 Score")
plt.grid(True, axis='y')
plt.show()

print(f"\nAverage CV F1-score: {np.mean(fold_scores):.4f}")

# 💾 Train final model
final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
final_clf.fit(X, y)
joblib.dump(final_clf, "final_model.pkl")
print("✅ Final model saved.")

# ⏳ Inference
df_test = pd.read_csv(test_csv)
test_paths = [os.path.join(test_img_dir, fname) for fname in df_test['image_id']]

# 📦 Reload ResNet18 for test feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet18.fc = torch.nn.Identity()
resnet18 = resnet18.to(device)
resnet18.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(img_paths):
    features = []
    for path in tqdm(img_paths, desc="Extracting test features"):
        img = Image.open(path).convert('RGB')
        img_tensor = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet18(img_tensor).cpu().numpy().flatten()
        features.append(feat)
    return np.array(features)

# 📷 Extract test features
X_test = extract_features(test_paths)

# 🔮 Predict
preds = final_clf.predict(X_test)
label_encoder_classes = np.load("label_encoder_classes.npy")
predicted_labels = label_encoder_classes[preds]

# 📤 Submission
df_submission = pd.DataFrame({
    'image_id': df_test['image_id'],
    'soil_type': predicted_labels
})
df_submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved: submission.csv")

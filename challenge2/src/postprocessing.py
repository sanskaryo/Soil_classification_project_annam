"""
Author: sanskar khandelwal
Team Name: TheLastTransformer
Team Members: 1
Leaderboard Rank: 56
"""


# postprocessing.py

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
import json

# Load saved features
train_features = np.load("train_features.npy")
test_features = np.load("test_features.npy")
test_ids = np.load("test_ids.npy", allow_pickle=True)

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Fit One-Class SVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
svm.fit(train_features)

# Predict on test set
svm_preds = svm.predict(test_features)
binary_preds = [1 if p == 1 else 0 for p in svm_preds]

# Save submission
submission = pd.DataFrame({
    'image_id': test_ids,
    'label': binary_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as 'submission.csv'.")

# Evaluate on training (positive) data
train_preds = svm.predict(train_features)
binary_train_preds = [1 if p == 1 else 0 for p in train_preds]
train_labels = [1] * len(binary_train_preds)
recall = recall_score(train_labels, binary_train_preds)
false_negatives = sum([1 for p in binary_train_preds if p == 0])

# Save evaluation metrics
metrics = {
    "model": "ResNet18 + One-Class SVM",
    "feature_dim": train_features.shape[1],
    "training_samples": len(train_labels),
    "nu": 0.1,
    "train_recall": recall,
    "false_negatives_est": false_negatives,
    "note": "Evaluated only on positive training data."
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("✅ metrics.json saved.")


#submission file is not included inrepo as it was not in the template
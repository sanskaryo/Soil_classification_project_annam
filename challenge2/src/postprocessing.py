import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

def tune_threshold(y_true, y_probs, thresholds=np.arange(0.1, 0.9, 0.01)):
    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        preds = (y_probs > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def prepare_submission(predictions, image_ids, threshold=0.5, output_file='submission.csv'):
    labels = (predictions > threshold).astype(int)
    submission = pd.DataFrame({
        'image_id': image_ids,
        'label': labels
    })
    submission.to_csv(output_file, index=False)
    print(f"[✅] Submission saved to '{output_file}' with threshold {threshold:.2f}")

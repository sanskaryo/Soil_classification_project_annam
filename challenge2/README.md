<h1 align="center">🌱 Soil Image Classification Challenge (Binary)</h1>

<p align="center">
  <img src="docs/cards/architecture.png" alt="Model Architecture" width="600" />
</p>

<p align="center">
  <strong>🔥 F1 Score: 0.8989 (Public) | Private Rank: 48</strong><br>
  🧠 Solo Participant | Finalist at <strong>Annam.ai × IIT Ropar</strong>
</p>

---

## 🧾 Overview

This repository contains my binary classifier solution for the Soil Image Classification Challenge organized by [Annam.ai](https://www.annam.ai/) and IIT Ropar. The goal is to predict whether an input image contains soil or not using CNN-based computer vision techniques.

**Why it matters:**
- 🌾 Precision agriculture & crop health
- ⛰️ Geological mapping & terrain analysis
- 🌍 Environmental monitoring & sustainability

---

## 📊 Leaderboard Performance

| Metric       | Public Score | Private Rank |
|--------------|--------------|--------------|
| 🔗 F1 Score  | 0.8989       | 48           |

---

## 🏁 Competition Details

| Detail          | Description                                 |
|-----------------|---------------------------------------------|
| **Organizer**   | Annam.ai × IIT Ropar                        |
| **Task**        | Binary classification (Soil / Non-Soil)      |
| **Deadline**    | May 25, 2025, 11:59 PM IST                  |
| **Evaluation**  | F1 Score (harmonic mean of Precision & Recall) |
| **Final Status**| Solo Submission, Finalist                   |

---

## 🧠 Model Pipeline

```mermaid
graph TD
  A[Raw Images] --> B[Preprocessing]
  B --> C[Train/Val Split + Augmentation]
  C --> D[Model (EfficientNet/Baseline)]
  D --> E[Inference]
  E --> F[Threshold Tuning]
  F --> G[Final Predictions CSV]
```

---

## 📁 Project Structure

```text
challenge2/
├── data/                # Dataset & synthetic 'Not Soil' images
├── docs/cards/          # Diagrams & cards
│   └── architecture.png # Model architecture
├── notebooks/           # Jupyter notebooks
│   ├── training.ipynb   # Model training workflow
│   └── inference.ipynb  # Inference & submission
├── src/                 # Processing scripts
│   ├── preprocessing.py # Data augmentation & synthetic images
│   └── postprocessing.py# Threshold tuning & metrics
├── download.sh          # Data download script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🏋️‍♂️ Training Highlights

- **Input Size:** 224×224 px
- **Augmentations:** RandomFlip, Rotation, ColorJitter
- **Model Architectures:** EfficientNet B0, ResNet variants
- **Loss Function:** Binary Cross-Entropy
- **Optimization:** Adam, learning rate scheduling

---

## 🧪 Evaluation & Thresholding

- **Metric:** Macro F1-Score
- **Threshold Tuning:** Grid search over [0.1, 0.9] to maximize validation F1.

```python
def tune_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    # evaluate F1 at each thresh... return best
```  

---

## 📌 Key Learnings

- **Data Augmentation** significantly improved generalization.  
- **EfficientNet** performed robustly despite class imbalance.  
- **Custom Thresholding** was crucial to boost F1 score.  
- **Modular Code** ensures reproducibility and ease of experimentation.  

---

## 🚀 Setup & Run

1. **Clone repo**
   ```cmd
   git clone https://github.com/your-username/soil-classification.git
   cd soil-classification/challenge2
   ```
2. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```
3. **Download data**
   ```cmd
   bash download.sh
   ```
4. **Prepare synthetic data**
   ```cmd
   python src/preprocessing.py
   ```
5. **Train model**
   - Open `notebooks/training.ipynb`, run all cells.
6. **Run inference**
   - Open `notebooks/inference.ipynb`, run all cells to generate `submission.csv`.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- **Annam.ai & IIT Ropar** for the challenge opportunity
- **PyTorch**, **Torchvision**, **OpenCV** for open-source tools
- **Kaggle community** for insights & kernels

---

## 🚀 Author

**Sanskar Khandelwal**  
2nd Year AIML Student, GLA University, Mathura  
✉️ sanskar.khandelwal_cs.aiml23@gla.ac.in
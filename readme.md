<h1>🏆 Soil Classification Challenge Submission</h1>

<p>This project was developed as part of the Hackathon + Internship opportunity organized by IIT Ropar and Annam.ai. I, Sanskar Khandelwal, participated solo and built ML models for classifying soil types from images. This task aimed to automate soil-type classification to assist in agriculture and sustainability using AI. Special thanks to <strong>Sudarshan Iyengar</strong>, <strong>Madhur Tharuja</strong>, and the entire <strong>Annam AI & IIT Ropar</strong> team for organizing this opportunity!</p>

---

## 👤 Participant Details

- **Name:** Sanskar Khandelwal  
- **Team Name:** solo_sanskar  
- **Year:** 2nd Year B.Tech CSE (AIML)  
- **University:** GLA University, Mathura  
- **Email:** sanskar.khandelwal_cs.aiml23@gla.ac.in  
- **Radhe Radhe! 🙏**  

---

## 📊 Leaderboard Performance

| Task | Score | Rank |
|------|-------|------|
| Task 1 - Binary Soil Classification | 1.000 | 56 |
| Task 2 - Multi-Class Soil Image Classification | 0.8989 | 37 |

---

## 🗂️ Project Structure

```bash
.
├── notebooks/
│   ├── training.ipynb         # Training pipeline for CNN classifier
│   └── inference.ipynb        # Generates predictions and submission CSV
├── models/                    # Saved model weights
├── data/                      # Preprocessed dataset
├── requirements.txt           # Python dependencies
├── download.sh                # Dataset download script
├── submission.csv             # Final submission predictions
└── README.md                  # This file
```

---

## 🧠 Approach Overview

### 🔹 Task Objective

Classify soil images into one of the four categories:  
- Alluvial  
- Black  
- Clay  
- Red

### 🔹 Modeling Pipeline

- **Model Architecture:** Transfer learning using pretrained CNNs like ResNet-18, EfficientNet-B0
- **Training Strategy:**  
  - Image normalization, resizing to 224x224  
  - Stratified train-validation split  
  - Data augmentation (flip, rotate, brightness)  
  - Cross-validation for robustness  
- **Inference:**  
  - Test-Time Augmentation (TTA)  
  - Ensemble averaging for stability  

---

## 🛠️ Tools & Technologies

- Python 🐍  
- PyTorch / Torchvision  
- Scikit-learn  
- OpenCV  
- Matplotlib / Seaborn  
- Jupyter Notebooks

---

## 📓 Notebooks Breakdown

### `training.ipynb`

- Loads and preprocesses image dataset
- Applies augmentations and normalizations
- Extracts features using pretrained CNNs (e.g., ResNet18)
- Trains classifiers (e.g., fully connected layers or Random Forests)
- Plots metrics and saves trained models

### `inference.ipynb`

- Loads saved models and test data
- Applies TTA (horizontal/vertical flips, brightness)
- Generates predictions
- Outputs `submission.csv` as per competition format

---

## 📈 Evaluation Metric

- **Metric Used:** Minimum F1-score across all 4 classes  
- This ensures balanced performance — even the lowest performing class matters!

```python
from sklearn.metrics import f1_score
score = min([
    f1_score(y_true, y_pred, average=None)[i] for i in range(4)
])
```

---

## ⚙ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/soil-classification
cd soil-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
bash download.sh
```

4. **Run notebooks**
- `notebooks/training.ipynb` → train models  
- `notebooks/inference.ipynb` → generate `submission.csv`

---

## ⚡ Why This Approach Works

✅ Combines deep learning feature extraction with classical ML models  
✅ Test-Time Augmentation for robust predictions  
✅ Balanced F1-score strategy ensures no class is ignored  
✅ Simple yet effective – reproducible and scalable  

---

## 💬 Reflections

I participated solo in this challenge and acknowledge that my submission may not compete head-to-head with full teams, but I gave my best and learned a lot! Looking forward to the next rounds if selected. Jai Shree Krishna 🙏

---

## 🤝 Acknowledgements

- Organizers: Annam.ai, IIT Ropar  
- Pretrained models: PyTorch Model Zoo  
- Community support and dataset providers  
- Inspiration from top teams and peers in this domain

---

## 👨‍💻 Author

**Sanskar Khandelwal**  
Email: `sanskar.khandelwal_cs.aiml23@gla.ac.in`  
University: GLA University, Mathura  
Connect with me for ML, AI, or vision projects! 🚀

---

## 📬 Contact

If any reviewer or peer wants to discuss this submission or connect:
- **Email:** sanskar.khandelwal_cs.aiml23@gla.ac.in

---

## ⚖️ License

This project is submitted as part of a Hackathon and is intended for academic and educational review. Please contact me for further use.

---

<p align="center"><strong>🚜 Towards Sustainable AI-Powered Agriculture! 🚀</strong></p>

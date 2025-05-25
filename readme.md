<h1> 🌱 Soil & Non-Soil Classification | Annam.ai @ IIT Ropar </h1>

<p>
This repository documents my solo submissions for both tasks in the AI + Internship challenge organized by <strong>Annam.ai</strong> in collaboration with <strong>IIT Ropar</strong>. The goal of this challenge was to classify soil types from images and distinguish soil vs non-soil samples using deep learning. I participated as an independent undergraduate and put my heart into solving both tasks with resourceful and explainable ML solutions.
</p>

---

## 👤 Participant Details

- **Name:** Sanskar Khandelwal  
- **Email:** sanskar.khandelwal_cs.aiml23@gla.ac.in  
- **College:** GLA University, Mathura  
- **Course:** B.Tech, CSE AIML (2nd Year)  
- **Team:** Solo participant  
- **Radhe Radhe 🙏**

---

## 🏅 Leaderboard Performance

| Task | Rank | Score |
|------|------|--------|
| Soil / Non-Soil Classification | 37 | 0.8989 |
| Soil Type Classification (Alluvial, Black, Clay, Red) | 56 | 1.000 |

---

## 🙏 Special Thanks

A heartfelt thank you to:

- **Sudarshan Iyengar**
- **Madhur Tharuja**
- The entire **Annam.ai & IIT Ropar team**

Apologies in case my submission felt incomplete compared to team entries — I was a solo participant. Still, I learned a lot and look forward to next rounds if selected!

---

## 🗂️ Repository Structure

```bash
.
├── soil_type/
│   ├── training.ipynb      # Training notebook for 4-class soil classification
│   └── inference.ipynb     # Inference + submission notebook
│
├── binary_soil/
│   ├── training.ipynb      # Soil vs Non-Soil model training
│   └── inference.ipynb     # Inference for binary classification
│
├── submission/
│   ├── soil_type_submission.csv
│   └── binary_submission.csv
│
├── requirements.txt        # Required libraries
├── download_data.sh        # Dataset download automation
└── README.md               # This file

```

🧠 Problem Statements
Task 1: Classify an image as either Soil or Non-Soil (binary classification)

Task 2: Classify each image into one of four soil types:

Alluvial

Black

Clay

Red

🧠 Modeling Approach
✅ Common Preprocessing
Image resizing to 224x224

RGB normalization (mean/std as per ImageNet)

Data augmentation (flip, rotation, color jitter)

Stratified data split for validation consistency

🔍 Feature Extraction
Used pretrained ResNet-18 as base CNN

Extracted 512-D embeddings from penultimate layer

Trained classic Random Forest on top of embeddings

🧪 Model Training
Used Stratified K-Fold Cross Validation (K=5)

Saved best models based on F1 score

Test-Time Augmentation (TTA) during prediction

📈 Evaluation Metric
🔥 Minimum F1-Score across all classes (for Task 2)
This metric ensures balanced performance across all categories by using the worst-performing class as the final score.

📓 Notebooks
training.ipynb
Load and preprocess training images

Extract deep features using pretrained ResNet

Train classifiers (Random Forest / Logistic Regression)

Save best models and logs

inference.ipynb
Load saved models and test images

Apply TTA for robustness

Save predictions in submission-ready CSV format

⚙️ How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/sanskarofficial/soil-annam.git
cd soil-annam
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download dataset:

bash
Copy
Edit
bash download_data.sh
Run training:

bash
Copy
Edit
# For Soil vs Non-Soil
cd binary_soil
jupyter notebook training.ipynb

# For Soil Type Classification
cd ../soil_type
jupyter notebook training.ipynb
Generate Predictions:

bash
Copy
Edit
jupyter notebook inference.ipynb
✨ Highlights
💪 Solo effort by 2nd-year undergrad

📊 Strong F1 scores on both tasks

🔁 Test-Time Augmentation improved generalization

🧠 Classic ML + Deep feature fusion worked better than end-to-end training (less overfit)

🤝 Acknowledgements
Annam.ai & IIT Ropar for organizing the challenge

PyTorch team for ResNet backbone

Fellow competitors and open source inspirations

🙋‍♂️ Author
Sanskar Khandelwal
Email: sanskar.khandelwal_cs.aiml23@gla.ac.in
Radhe Radhe 🙏

📜 License
This project is under the MIT License – feel free to use, modify, or extend it for educational and research purposes.

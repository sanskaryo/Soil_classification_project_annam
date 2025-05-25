<h1 align="center">🌱 Soil Image Classification Challenge</h1>

<p align="center">
  <img src="docs/cards/architecture.png" alt="Model Architecture" width="500"/>
</p>

<p align="center">
  <b>🔥 F1 Score (Public Leaderboard): 0.8989</b><br>
  🧠 Solo Participant | 🎯 Finalist in <b>Soil Classification Challenge</b> by Annam.ai @ IIT Ropar
</p>

---

## 🧾 Overview

This project is my solution to the **Soil Image Classification Challenge** organized by [Annam.ai](https://www.annam.ai/) at **IIT Ropar**.  
The task: **Classify whether an image contains soil or not**.

🪴 Soil classification is crucial for:
- Agriculture & crop health 🌾
- Geology & terrain mapping ⛰️
- Environmental impact analysis 🌍

> **Goal:** Train a binary classifier on images and predict whether it contains soil using CNN-based computer vision models.

---

## 🏁 Competition Details

| Detail              | Info                                      |
|---------------------|-------------------------------------------|
| **Organizer**       | IIT Ropar × Annam.ai                      |
| **Task**            | Binary Image Classification (Soil/Non-Soil) |
| **Deadline**        | May 25, 2025, 11:59 PM IST                |
| **Metric**          | 🔁 F1 Score (harmonic mean of Precision & Recall) |
| **Score Achieved**  | 💯 0.8989 (Public LB)                     |
| **Status**          | ✅ (Solo Submission)             |

---

## 🧠 Model Pipeline

```mermaid
graph TD
A[Input Raw Images] --> B[Preprocessing]
B --> C[Train/Val Split + Augmentation]
C --> D[Model: CNN (EfficientNet/Baseline)]
D --> E[Inference on Test Images]
E --> F[Postprocessing & Threshold Tuning]
F --> G[Final Predictions CSV]


## 📁 Project Structure

```text
soil-classification-project/
│
├── data/                   # Empty, gets populated via script
├── docs/cards/             # Plots, model diagrams & notes
│   └── architecture.png    # Architecture diagram (to generate)
├── notebooks/              # Primary Jupyter notebooks
│   ├── training.ipynb      # Model training workflow
│   └── inference.ipynb     # Test-set predictions
├── src/                    # Core Python modules
│   ├── preprocessing.py    # Data prep & synthetic image generation
│   └── postprocessing.py   # Threshold tuning & metrics
├── download.sh             # Kaggle data download script
├── README.md               # Project overview & instructions
├── LICENSE                 # MIT License
├── requirements.txt        # Python dependencies
├── ml-metrics.json         # Final metric JSON report
└── .gitignore              # Exclude caches & checkpoints
```



🏋️‍♂️ Training Highlights
📏 Input size: 224x224 (resized)

🎨 Augmentations: RandomFlip, Rotation, ColorJitter

🧠 Model used: EfficientNet / ResNet (based on experiment)

🔍 Loss: Binary Cross Entropy

🧪 Eval: F1-Score (macro + threshold tuning)



🧪 Evaluation & Thresholding
Final threshold is tuned using validation F1-score.

python
Copy
Edit
def tune_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    ...
🧰 Usage
## 🚀 Setup & Run

Follow these steps to get started locally:

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/soil-classification-project.git
   cd soil-classification-project
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**

   ```bash
   bash download.sh  # run in bash/WSL/Git Bash on Windows
   ```

4. **Generate synthetic 'Not Soil' images**

   ```bash
   python src/preprocessing.py
   ```

5. **Train the model**

   Open `notebooks/training.ipynb` in Jupyter and run all cells to train and validate.

6. **Run Inference**

   Open `notebooks/inference.ipynb` in Jupyter and execute all cells to generate final predictions.
📊 Result
Metric	Score
F1 Score	0.8989
Precision	~0.91
Recall	~0.89
Model Size	~20MB
Threshold	Tuned @ 0.45

📌 Key Learnings
Data augmentations made a huge difference 🎯

EfficientNet generalised well despite class imbalance

Custom thresholding = clutch move to boost F1 📈

Clear code = better reproducibility + judging score

📜 License
This repository is licensed under the MIT License.

🙌 Acknowledgements
Special thanks to:

IIT Ropar × Annam.ai for organizing the competition

OpenCV, PyTorch, torchvision for open-source magic

The Kaggle community for 💎 helpful kernels & ideas

🚀 Author
Sanskar – 2nd Year AIML Student
📍 GLA University | 👨‍💻 Building cool ML tools
🔗 email - sanskar.khandelwal_cs.aiml23@gla.ac.in
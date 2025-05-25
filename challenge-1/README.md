#  Soil Classification Hackathon Challenge-1

This project performs soil type classification using a deep learning model trained on soil spectral or feature data. The model is developed in PyTorch and evaluated with metrics such as Accuracy and F1-score.

---

##  Dataset

We use the dataset provided through the [Kaggle Competition: Soil Classification](https://www.kaggle.com/competitions/soil-classification).

To download the dataset, use the provided shell script:

```bash
bash download.sh
```
##  Model Details

- **Framework**: PyTorch  
- **Backbone**: ResNet18-based neural network  
- **Input**: Numerical soil features  
- **Output**: Soil class prediction  
- **Current F1-score**: **0.96**

### 1. Clone the Repository

```bash
git clone https://github.com/nav-jk/soil_classification_1_annam
cd soil_classification_1_annam
```
### 2. Model Training

Open and run the notebook:

```markdown
training.ipynb
```
### 3.Inference

To make predictions on new soil data, open and run:

```markdown
inference.ipynb
```
##  Model Insight

### Overview

The soil classification model is built using a **ResNet18-based neural network** implemented in PyTorch. ResNet (Residual Network) is a deep convolutional neural network architecture designed to tackle the vanishing gradient problem by introducing residual connections. These connections allow gradients to flow more easily through deep networks during training, enabling the network to learn more complex patterns effectively.

### Why ResNet18?

ResNet18 strikes a good balance between depth and computational efficiency. Its relatively shallow architecture (compared to deeper ResNet variants) allows faster training and inference while still maintaining strong feature extraction capabilities. This makes it ideal for a tabular or spectral dataset where input features are numerical soil parameters rather than raw images.

### Input and Feature Engineering

The model takes numerical soil features as input — typically soil spectral measurements or other relevant physical/chemical properties. Proper preprocessing ensures that features are normalized and scaled appropriately, which helps the model converge faster and achieve better accuracy.

### Model Architecture

- The backbone is adapted from ResNet18, with modifications to suit the feature dimension and output classes.
- The final fully connected layer outputs class probabilities corresponding to different soil types.
- Activation functions like ReLU are used to introduce non-linearity, enabling the model to capture complex soil patterns.
- Dropout or batch normalization layers may be used to improve generalization and prevent overfitting.

### Training Process

- The model is trained using cross-entropy loss, a standard for multi-class classification.
- Optimization is typically done using Adam or SGD optimizers with carefully tuned learning rates.
- The training loop includes regular validation to monitor overfitting and guide hyperparameter tuning.
- Early stopping or model checkpoints save the best-performing model weights.

### Evaluation Metrics

- Accuracy provides a straightforward measure of correct predictions.
- The F1-score, particularly important in imbalanced classification problems, balances precision and recall to give a more holistic performance measure.
- Achieving an F1-score of **0.96** indicates excellent model capability in distinguishing soil classes.



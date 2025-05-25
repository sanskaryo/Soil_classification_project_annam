import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os


DATA_DIR = "data/train"
TEST_DIR = "data/test"
TRAIN_LABELS_CSV = "data/train_labels.csv"
TEST_IDS_CSV = "data/test_ids.csv"
BATCH_SIZE = 32
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LEARNING_RATE = 1e-4
CLASSES = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']

label2idx = {label: idx for idx, label in enumerate(CLASSES)}


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = val_transform


class SoilDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_csv)

        #  Shuffle rows to prevent class ordering bias
        self.labels_df = self.labels_df.sample(frac=1, random_state=42).reset_index(drop=True)

        self.transform = transform
        self.labels_df['label_idx'] = self.labels_df['soil_type'].map(label2idx)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['image_id']
        label = self.labels_df.iloc[idx]['label_idx']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


full_dataset = SoilDataset(DATA_DIR, TRAIN_LABELS_CSV, transform=train_transform)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    class_f1 = f1_score(all_labels, all_preds, average=None)
    return epoch_loss, epoch_f1, class_f1


best_min_class_f1 = 0

for epoch in range(EPOCHS):
    train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_f1, val_class_f1 = validate(model, val_loader, criterion, DEVICE)

    min_class_f1 = val_class_f1.min()

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Macro F1: {val_f1:.4f}")
    print(f"Per-Class F1: {val_class_f1}")
    print(f"Min Class F1: {min_class_f1:.4f}")

    if min_class_f1 > best_min_class_f1:
        best_min_class_f1 = min_class_f1
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")


model.load_state_dict(torch.load("best_model.pth"))
model.eval()


class TestSoilDataset(Dataset):
    def __init__(self, img_dir, test_ids_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.test_ids_df = pd.read_csv(test_ids_csv)
        self.image_ids = self.test_ids_df['image_id'].tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

test_dataset = TestSoilDataset(TEST_DIR, TEST_IDS_CSV, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


predictions = []
with torch.no_grad():
    for images, image_ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        for img_id, pred in zip(image_ids, preds):
            predictions.append((img_id, CLASSES[pred]))


submission_df = pd.DataFrame(predictions, columns=["image_id", "soil_type"])
submission_df.to_csv("submission1.csv", index=False)

print("Submission saved to submission.csv")

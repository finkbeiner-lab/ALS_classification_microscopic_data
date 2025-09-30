import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from maxvit import *
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


TRAIN_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_Sporadic/train" 
VAL_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_Sporadic/val"
TEST_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_Sporadic/test"


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
	
        torch.save(model, "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/maxvit_model_full.pth")
    return model

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct / total
    avg_loss = test_loss / len(test_loader)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return all_preds, all_labels

# ------------------------------
# Example usage after training
# ------------------------------
# Assume you have a test_loader similar to train_loader/val_loader
#all_preds, all_labels = test_model(model, test_loader, criterion, device)





# ------------------------------
# Data transforms
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((448, 448)),  # MaxViT input size
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # adjust if RGB
])

# Example: ImageFolder dataset (organize as train/live, train/dead, val/live, val/dead)
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_PATH, transform=transform)
test_dataset   = datasets.ImageFolder(TEST_PATH, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader   = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)



model = MaxViT(tiny_args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------------
# Loss & Optimizer
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4 )


model = train(model, train_loader, val_loader, criterion, optimizer, device, 10)


all_preds, all_labels = test_model(model, test_loader, criterion, device)



# all_preds, all_labels are lists or numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
disp.plot(cmap=plt.cm.Blues)  # you can choose any colormap
plt.title("Confusion Matrix - Test Data")
plt.savefig("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/confusion_matrix.png")









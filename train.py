import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# Step 1: Define custom Dataset for .tif images
class TIFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: directory with subfolders per class, e.g.,
        root_dir/
            class1/
                img1.tif
                ...
            class2/
                img2.tif
                ...
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_folder = self.root_dir / cls_name
            for img_path in cls_folder.glob("*.tif"):
                self.image_paths.append(img_path)
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = tifffile.imread(path)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)  # Grayscale to RGB

        image = image.astype(np.float32)
        image /= image.max()

        image = Image.fromarray((image * 255).astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
    
# Step 2: DINOv2 Transform (224x224 expected size)
dinov2_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])
])


# Step 4: Define DINOv2-based classifier
class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")  # or "DEFAULT" in newer versions
        self.backbone.heads = nn.Identity()  # Remove DINO head
        self.classifier = nn.Sequential(
    nn.Linear(self.backbone.embed_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
        #self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            total_loss += loss.item() * len(y)
            correct += (pred.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)



# Step 3: Load Dataset
train_dataset = TIFDataset("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/train/", transform=dinov2_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = TIFDataset("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/val/", transform=dinov2_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

test_dataset = TIFDataset("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/test/", transform=dinov2_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Step 5: Initialize and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DINOv2Classifier(num_classes=len(train_dataset.class_to_idx)).to(device)

# Example training loop (1 epoch)

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
criterion = nn.CrossEntropyLoss()
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 75
train_losses = []
val_losses=[]
train_accuracies, val_accuracies = [], []
# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    #scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
}, "checkpoint_ep100_bs8_wd.pth")


test_loss, test_acc = eval_model(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
torch.save

# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()


plt.savefig("loss_accuracy_curve_ep100.png")





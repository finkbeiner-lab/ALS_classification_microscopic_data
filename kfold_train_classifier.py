import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import timm
import pandas as pd
import os
from glob import glob
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from transformers import ViTModel, ViTConfig
from models.vit_simclr import ViTSimCLR
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
import plotly.graph_objects as go
import random


def plot_curve(train_losses,val_losses,epochs, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1,1+epochs), train_losses, 'bo-', label='Training Loss')
    plt.plot(np.arange(1,1+epochs), val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Save the plot to a file
    plt.savefig(save_path, dpi=300)  # or .pdf, .svg, etc.

np.random.seed(0)

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
    
"""   
class ViTSimCLR(nn.Module):

    def __init__(self, out_dim):
        super(ViTSimCLR, self).__init__()
        self.encoder = ViTModel.from_pretrained("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/vit-base-patch16-224-in21k")
        #self.feature_dim = self.encoder.config.hidden_size  # usually 768
        #vitEncoder = self.encoder
        #print(vitEncoder)
        num_ftrs = self.encoder.config.hidden_size

        #self.features = nn.Sequential(*list(vitEncoder.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)


    def forward(self, x):
        outputs = self.encoder(x)  # h: (batch_size, feature_dim), from [CLS] token
        h = outputs.last_hidden_state[:, 0]       # CLS token (B, hidden_dim)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x  # h: representation, x: projection
"""   
    


    
    
    
    
class TIFDataset_kfold(Dataset):
    def __init__(self, root_dir1, root_dir2, transform=None):
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
        self.root_dir1= Path(root_dir1)
        self.root_dir2 = Path(root_dir2)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        #self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        dir_list1 = [x for x in os.listdir(root_dir1) if x!='.DS_Store']
        dir_list2 = [x for x in os.listdir(root_dir2) if x!='.DS_Store']
        self.class_to_idx1 = {cls_name: idx for idx, cls_name in enumerate(sorted(dir_list1))}
        self.class_to_idx2 = {cls_name: idx for idx, cls_name in enumerate(sorted(dir_list2))}

        #print(self.class_to_idx)
        for cls_name, cls_idx in self.class_to_idx1.items():
            cls_folder = self.root_dir1 / cls_name
            for img_path in cls_folder.glob("*.tif"):
                self.image_paths.append(img_path)
                self.labels.append(cls_idx)
        
        for cls_name, cls_idx in self.class_to_idx2.items():
            cls_folder = self.root_dir2 / cls_name
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



class SimCLRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SimCLRClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512), #old 2048
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features, _ = self.encoder(x)
        out = self.classifier(features)
        return out
    
def load_pretrained_simclr_model(model_path):
    model = ResNetSimCLR(base_model= "resnet50",out_dim=2 )# .to(self.device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_pretrained_vitsimclr_model(model_path):
    model = ViTSimCLR()# .to(self.device)
    print(model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model




def evaluate_model(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for i, (batch_imgs, batch_labels) in enumerate(val_loader):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        batch_size = batch_imgs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    avg_loss = total_loss / total_samples
    return avg_loss


def train_classifier(train_loader, val_loader, device, simclr_model, num_classes, num_epochs, model_checkpoints_folder):
    model = SimCLRClassifier(simclr_model, num_classes=num_classes)
    model.to(device)
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    running_loss = 0.0
    total_samples = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10  # number of epochs to wait
    wait = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (batch_imgs, batch_labels) in enumerate(train_loader):
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            #print(i, batch_imgs.shape, batch_labels.shape, outputs.shape)
            loss = criterion(outputs, batch_labels)
            #print("loss",loss)
            loss.backward()
            optimizer.step()
            #break
            
            #print("idx", i, "loss", loss.item())
            batch_size = batch_imgs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
        
        scheduler.step()
        avg_loss = running_loss / total_samples
        train_losses.append(avg_loss)
        val_loss  = evaluate_model(model, device, val_loader,criterion)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss:{val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'using_resnet1_model_train_val_new.pth'))
        print('saved')
        """
            # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_state = model.state_dict()  # save best model
        else:
            wait += 1
            print(f"  ↳ No improvement for {wait}/{patience} epochs.")
            if wait >= patience:
                print("Early stopping triggered.")
                break
        """
    return model, train_losses, val_losses


def val_classifier(val_loader, device,model,class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_preds2 = []
    correct, total = 0, 0
    with torch.no_grad():
        for i, (batch_imgs, batch_labels) in enumerate(val_loader):
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            outputs = model(batch_imgs)
            probs = F.softmax(outputs, dim=1)[:,1]
            y_pred =  outputs.argmax(axis=1)
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_preds2.extend(y_pred.detach().cpu().numpy())
            correct += (y_pred == batch_labels).sum().item()
            total += batch_labels.size(0)
    # Report
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(report)
    #report.to_csv('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/report.csv')
    acc = 100.0 * correct / total
    return acc, all_preds2, all_labels


    
if __name__ == '__main__':
    #model_path  = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Jul29_11-10-57_DDLUFB08808/checkpoints/model.pth"
    #model_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Aug06_20-30-08_kif-gh200-01.gladstone.internal/checkpoints/model.pth"
    model_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Aug08_18-10-56_kif-gh200-01.gladstone.internal/checkpoints/model.pth"
    input_shape = (224,224,3)
    batch_size = 8
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225]),
])
    
    train_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/train/"
    val_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/val/"
    
    train_dataset = TIFDataset_kfold(train_dir,val_dir, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #test_dataset = TIFDataset("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/test/", transform=data_transforms)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #train_loader = get_train_data_loaders(train_dataset, batch_size,num_workers)
    #valid_loader  = get_val_data_loaders(val_dataset, batch_size,num_workers)

    #train_loader, valid_loader = get_train_validation_data_loaders(train_dataset, batch_size, num_workers, valid_size)
    
    #simclr_model = load_pretrained_vitsimclr_model(model_path)
    
    k_folds = 3
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    num_classes = 2
    num_epochs = 60
    model_checkpoints_folder = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/saved_models'
    class_names = ["0","1"]
    results = {}
    
    fig = go.Figure()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        
        simclr_model = load_pretrained_simclr_model(model_path)
        
        print(f"\nFOLD {fold}")
        print("-----------------------------")

        # Subset datasets for this fold
        train_subsampler = Subset(train_dataset, train_idx)
        val_subsampler = Subset(train_dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subsampler, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subsampler, batch_size=8, shuffle=False)
        
        model, train_losses, val_losses = train_classifier(train_loader, val_loader, device, simclr_model, num_classes, num_epochs, model_checkpoints_folder)
    
        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'using_kfold_resnet1_model_train_val_new.pth'))
        print('final saved')
     
        acc, all_preds, all_labels = val_classifier(val_loader, device,model,class_names)
        print(f"Accuracy for fold {fold}: {acc:.2f} %")
        results[fold] = acc
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(all_labels,all_preds, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Create ROC curve with Plotly
        

        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            #line=dict(color='blue', width=2)
        ))

    # Add diagonal line for random guessing
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        width=700,
        height=500
)
    fig.write_html("kfold_roc_curve.html")
        
        
    
    #val_classifier(test_loader, device,model,class_names)
    
    save_path =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/kfold_model_train_val_new_loss_curve.png'
    plot_curve(train_losses,val_losses,len(train_losses), save_path)
    
    
    
    


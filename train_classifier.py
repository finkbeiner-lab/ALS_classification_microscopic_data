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
from datetime import datetime


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
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),    #, norm_layer=nn.InstanceNorm2d),
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

        #self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        dir_list = [x for x in os.listdir(root_dir) if x!='.DS_Store']
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(dir_list))}
        #print(self.class_to_idx)
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



class SimCLRClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SimCLRClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(512, 224), #old 2048 for resnet 50, 512 for resnet 18
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(224, num_classes)
        )

    def forward(self, x):
        features, _ = self.encoder(x)
        out = self.classifier(features)
        return out
    
def load_pretrained_simclr_model(model_path,base_model= "resnet50"):
    model = ResNetSimCLR(base_model=base_model,out_dim=2 )# .to(self.device)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    running_loss = 0.0
    total_samples = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5  # number of epochs to wait
    wait = 0
    for epoch in range(num_epochs):
        model.train()
        for i, (batch_imgs, batch_labels) in enumerate(train_loader):
            batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            #print(outputs.shape, batch_labels.shape )
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            #break
            #print(i, batch_imgs.shape, batch_labels.shape, outputs.shape)
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
    for i, (batch_imgs, batch_labels) in enumerate(val_loader):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
        outputs = model(batch_imgs)
        y_pred =  outputs.argmax(axis=1)
        all_preds.extend(y_pred.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    # Report
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(report)
    #report.to_csv('/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/crop_classifier/report.csv')



    
if __name__ == '__main__':
    #model_path  = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Jul29_11-10-57_DDLUFB08808/checkpoints/model.pth"
    #model_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Aug06_20-30-08_kif-gh200-01.gladstone.internal/checkpoints/model.pth"
    #model_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Aug08_18-10-56_kif-gh200-01.gladstone.internal/checkpoints/model.pth" #-- TDP-43
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_Sporadic/runs/Aug27_23-04-34_kif-gh200-01.gladstone.internal/checkpoints/model.pth" -- ALL SPORADIC
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/runs/Aug28_04-11-23_kif-gh200-02.gladstone.internal/checkpoints/model.pth"
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/GSGT_T6/runs/Aug29_05-31-02_kif-gh200-01.gladstone.internal/checkpoints/model.pth"
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/GSGT_T6/runs/Sep05_10-18-04_kif-gh200-04.gladstone.internal/checkpoints/model.pth"
    #MODEL_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/runs/Sep17_11-58-45_kif-gh200-02.gladstone.internal/checkpoints/model.pth"
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_Sporadic/runs/Sep19_12-43-04_kif-gh200-03.gladstone.internal/checkpoints/model.pth" 
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Oct28_11-10-49_kif-gh200-02.gladstone.internal/checkpoints/model.pth" 
    
    #MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Oct28_11-10-49_kif-gh200-02.gladstone.internal/checkpoints/model.pth" 
    MODEL_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/runs/Dec04_22-44-11_kif-gh200-01.gladstone.internal/checkpoints/model.pth"
    

    #TRAIN_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/train/" --TDP 43
    #TRAIN_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_Sporadic/train/"
    TRAIN_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/C9ORF72_T6/train/"
    #TRAIN_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/train/"
    #VAL_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/val/"
    VAL_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/C9ORF72_T6/val/"    
    #TEST_PATH =  "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/test/"
    TEST_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/C9ORF72_T6/test/"
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
    model_checkpoints_folder = '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/C9ORF72_T6/saved_models/'+str(formatted_datetime)
    os.makedirs(model_checkpoints_folder)
    #plot_save_path =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/using_resnet1_model_train_val_new_loss_curve.png'
    #plot_save_path =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/All_C9ORF72/Results/'+ MODEL_PATH.split("/")[-3]+'_loss_curve.png'
    plot_save_path =  '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/C9ORF72_T6/Results/'+ MODEL_PATH.split("/")[-3]+'_loss_curve.png'
    input_shape = (224,224,3)
    batch_size = 4
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])])
    
    train_dataset = TIFDataset(TRAIN_PATH, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = TIFDataset(VAL_PATH, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = TIFDataset(TEST_PATH, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #train_loader = get_train_data_loaders(train_dataset, batch_size,num_workers)
    #valid_loader  = get_val_data_loaders(val_dataset, batch_size,num_workers)

    #train_loader, valid_loader = get_train_validation_data_loaders(train_dataset, batch_size, num_workers, valid_size)
    simclr_model = load_pretrained_simclr_model(MODEL_PATH, "resnet18")
    #simclr_model = load_pretrained_vitsimclr_model(model_path)
    
    num_classes = 2
    num_epochs = 1
    
    model, train_losses, val_losses = train_classifier(train_loader, val_loader, device, simclr_model, num_classes, num_epochs, model_checkpoints_folder)
    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
    print('final saved')
    
    class_names = ["0","1"]
    val_classifier(val_loader, device,model,class_names)
    val_classifier(test_loader, device,model,class_names)
    
    plot_curve(train_losses,val_losses,len(train_losses), plot_save_path)
    
    
    
    


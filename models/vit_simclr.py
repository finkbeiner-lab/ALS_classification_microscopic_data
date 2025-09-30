import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # install with `pip install timm`
from transformers import ViTModel, ViTConfig


class ViTSimCLR(nn.Module):
    def __init__(self, base_model='vit_base_patch16_224', out_dim=2, pretrained=True):
        super(ViTSimCLR, self).__init__()

        # Load ViT backbone without classifier head
        #self.vit = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        self.encoder = ViTModel.from_pretrained("/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Julia_TDP43/CODES/vit-base-patch16-224-in21k")
        self.feature_dim = self.encoder.config.hidden_size  # usually 768

        # Get the embedding dimension (e.g., 768 for ViT-Base)
        #self.feature_dim = self.vit.num_features

        # Projection MLP (same as in SimCLR)
        self.l1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.l2 = nn.Linear(self.feature_dim, out_dim)

    def forward(self, x):
        outputs = self.encoder(x)  # h: (batch_size, feature_dim), from [CLS] token
        h = outputs.last_hidden_state[:, 0]       # CLS token (B, hidden_dim)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x  # h: representation, x: projection

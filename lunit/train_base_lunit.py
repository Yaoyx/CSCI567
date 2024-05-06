import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(13)

class_ind = {'CC':0, 'EC':1, 'LGSC':2, 'HGSC':3, 'MC':4}

class OvarianImages(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.img_metadata = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        # patch_idx = int(np.floor(idx / len(self.img_metadata)))
        sample_idx = idx % len(self.img_metadata) 
        sample = self.img_metadata.iloc[sample_idx, 0]

        label = torch.tensor(class_ind[self.img_metadata.iloc[sample_idx, 1]]).to(device)  

        image_patches = []
        for i in range(100):
            img_path = self.img_dir + f'sample_{sample}' + f'/{sample}_{i}.png'
            patch = torch.tensor(np.asarray(Image.open(img_path))[13:237, 13:237].T, dtype=torch.float32)
            image_patches.append(patch)
        image_patches = torch.stack(image_patches, dim=0).to(device)
        return image_patches, label
    

# Create Dataset
metadata = "/scratch1/yuqiuwan/CSCI567/train.csv"
image_dir = "/scratch1/yuqiuwan/CSCI567/mask_images/"
imgs = OvarianImages(metadata, image_dir)
train_set, test_set = torch.utils.data.random_split(imgs, [0.8, 0.2])

# Base model
class BaselineModel(nn.Module):
    def __init__(self, FeatureExtractor, d_embed=768, n_classes=5):
        super().__init__()
        self.image_encoder = FeatureExtractor
        self.image_proj = nn.Linear(d_embed, n_classes)
        self.n_classes = n_classes

    def forward(self, image):
        features = self.image_encoder(image)

        logits = self.image_proj(features).view(-1, 100, self.n_classes)
        logits = torch.sum(logits, axis=1)

        return logits
    


import torch
from torchvision.models.resnet import Bottleneck, ResNet

import torch
from timm.models.vision_transformer import VisionTransformer


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model


from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTModel



def train_loop(dataloader, model, loss_fn, optimizer, image_processor=None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (imgs, labels) in enumerate(dataloader): 
        imgs = imgs.view(-1, 3, 224, 224)
        if image_processor:
            imgs = image_processor(imgs, return_tensors="pt").to(device)
            logits = model(imgs['pixel_values'])
        else:
            # Compute prediction and loss
            logits = model(imgs)
        loss = loss_fn(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(torch.argmax(logits, axis=1) == labels) / logits.shape[0]

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(f"loss: {loss:>7f}; train_acc: {train_acc:>7f}  [{current:>5d}/{size:>5d}]")

######################## Train Base Model ########################
batch_size = 2
dropout = 0.0
learning_rate = 10**(-3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# load phikon
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
extractor = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
extractor.eval()

base_model = BaselineModel(extractor, d_embed=384, n_classes=5)
base_model.to(device)
# Freeze the pre-trained model
for p in base_model.image_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, base_model, F.cross_entropy, optimizer, image_processor)
print("Done!")

torch.save(base_model.state_dict(), '/scratch1/yuqiuwan/CSCI567/trainedModels/lunit_base_mask.pt')
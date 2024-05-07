import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from transformers import AutoImageProcessor, ViTModel

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
            patch = torch.tensor(cv2.imread(img_path)[13:237, 13:237].T, dtype=torch.float32)
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

    def forward(self, *args, **kwargs):
        outputs = self.image_encoder(*args, **kwargs)
        #print(outputs)
        features = outputs.cuda()
        #print(features)
        logits = self.image_proj(features).view(-1, 100, self.n_classes)
        logits = torch.sum(logits, axis=1)

        return logits
    

def train_loop(dataloader, model, loss_fn, optimizer, image_processor=None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (imgs, labels) in enumerate(dataloader): 
        imgs = imgs.view(-1, 3, 224, 224)
        if image_processor:
            imgs = image_processor(imgs, return_tensors="pt").to(device)
            # Compute prediction and loss
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

# load ctranspath
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
extractor = ctranspath()
extractor.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
extractor.load_state_dict(td['model'], strict=True)
extractor.eval()

# Initialize base model
base_model = BaselineModel(extractor, d_embed=768, n_classes=5)
base_model.to(device)
# Freeze the pre-trained model
for p in base_model.image_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)


epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, base_model, F.cross_entropy, optimizer, image_processor=image_processor)
print("Done!")


torch.save(base_model.state_dict(), '/scratch1/yuqiuwan/CSCI567/ctranspath_base_mask.pt')
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
import torchmetrics

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
image_dir = "/scratch1/yuqiuwan/CSCI567/preprocess_images_threshold/"
imgs = OvarianImages(metadata, image_dir)
train_set, test_set = torch.utils.data.random_split(imgs, [0.8, 0.2])


image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
extractor = ctranspath()
extractor.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
extractor.load_state_dict(td['model'], strict=True)
extractor.eval()

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
    
# Initialize base model
base_model = BaselineModel(extractor, d_embed=768, n_classes=5)
base_model.load_state_dict(torch.load('/scratch1/yuqiuwan/CSCI567/trainedModels/ctranspath_base.pt'))
base_model.eval()
base_model.to(device)

print(device)

from torchmetrics.classification import MulticlassAUROC
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

def test_loop(dataloader, model, image_processor=None):
    size = len(dataloader.dataset)
    model.eval()
    correct_num = 0

    preds_temp = []
    labels_temp = []
    with torch.no_grad(): 
        for batch, (imgs, labels) in enumerate(dataloader): 
            imgs = imgs.view(-1, 3, 224, 224)
            if image_processor:
                imgs = image_processor(imgs, return_tensors="pt").to(device)
                logits = model(imgs['pixel_values'])
            else:
                logits = model(imgs)
            preds_temp.append(logits)
            labels_temp.append(labels)

            correct_num += torch.sum(torch.argmax(logits, axis=1) == labels) 
            print(batch, 'correct_num:', correct_num)
    
    mypreds = torch.cat(preds_temp, dim=0)
    mylabels = torch.cat(labels_temp, dim=0)
    metric = MulticlassAUROC(num_classes=5, average=None, thresholds=None)
    print("AUC: ", metric(mypreds, mylabels))
    precision = torchmetrics.Precision(task="multiclass", num_classes=5, average="none").to(device)
    recall = torchmetrics.Recall(task="multiclass", num_classes=5, average="none").to(device)
    precision_score = precision(mypreds, mylabels)
    recall_score = recall(mypreds, mylabels)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=5, average="none").to(device)
    accuracy = acc(mypreds, mylabels)
    test_acc = correct_num  / size
    print('Test_accuracy:', test_acc)
    print('Precision:', precision_score)
    print('Recall:', recall_score)
    print("Accuracy: ", accuracy)

        
test_loop(test_dataloader, base_model, image_processor)
print("Done!")
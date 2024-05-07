import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoImageProcessor, ViTModel

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(13)

class_ind = {'CC':0, 'EC':1, 'LGSC':2, 'HGSC':3, 'MC':4}

class OvarianDataset(Dataset):
    def __init__(self, annotations_file, img_dir, text_dir):
        self.img_metadata = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.texts = {}
        for c in ['CC', 'EC', 'LGSC', 'HGSC', 'MC']:
            self.texts[c] = pd.read_table(text_dir+c+'.txt', header=None, sep='.').iloc[:,1].to_list()
       
    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        sample_idx = idx
        sample = self.img_metadata.iloc[sample_idx, 0]
        group = self.img_metadata.iloc[sample_idx, 1]

        label = torch.tensor(class_ind[group]).to(device)  

        image_patches = []
        for i in range(100):
            img_path = self.img_dir + f'sample_{sample}' + f'/{sample}_{i}.png'
            patch = torch.tensor(np.asarray(Image.open(img_path))[13:237, 13:237].T, dtype=torch.float32)
            image_patches.append(patch)
        image_patches = torch.stack(image_patches, dim=0).to(device)

        text = self.texts[group][idx % 100]
        # text = torch.tensor(self.tokenizer.encode(text, max_length=seq_max_length, padding="max_length")).to(device)

        return image_patches, text, label



# Create Dataset
metadata = "/scratch1/yuqiuwan/CSCI567/train.csv"
image_dir = "/scratch1/yuqiuwan/CSCI567/preprocess_images_threshold/"
text_dir = "/scratch1/yuqiuwan/CSCI567/textLabel/"

wholedataset = OvarianDataset(metadata, image_dir, text_dir)
train_set, test_set = torch.utils.data.random_split(wholedataset, [0.8, 0.2])

class CLIP(nn.Module):
    def __init__(self, ImageEncoder, TextEncoder, d_embed=[384,768], n_classes=5):
        super().__init__()
        self.image_encoder = ImageEncoder
        self.text_encoder = TextEncoder
        self.image_proj = nn.Linear(d_embed[0], n_classes)
        self.text_proj = nn.Linear(d_embed[1], n_classes)
        self.n_classes = n_classes

    def forward(self, img, text):
        img_features = self.image_encoder(img)
        img_embed = self.image_proj(img_features).view(-1, 100, self.n_classes)
        img_embed = torch.mean(img_embed, 1)
        img_embed = img_embed / torch.norm(img_embed, dim=-1, keepdim=True)

        text_outputs = self.text_encoder(**text)
        text_embed = text_outputs.pooler_output
        text_embed = self.text_proj(text_embed)
        text_embed = text_embed / torch.norm(text_embed, dim=-1, keepdim=True)

        logits = img_embed @ text_embed.T
        return logits
    

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


seq_max_length = 50
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

def contrastive_loss(logits, labels):
    image_loss = F.cross_entropy(logits, labels, reduction="mean")
    text_loss = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss = (image_loss + text_loss) / 2

    return loss

def train_loop(dataloader, model, loss_fn, optimizer, image_processor=None):
    size = len(dataloader.dataset)
    model.train()
    best_loss = np.inf
    for batch, (imgs, texts, _) in enumerate(dataloader):
        labels = torch.tensor(range(batch_size)).to(device)
        imgs = imgs.view(-1, 3, 224, 224)
        texts = tokenizer(texts, padding='max_length', max_length=seq_max_length, return_tensors='pt').to(device)

        if image_processor:
            imgs = image_processor(imgs, return_tensors="pt").to(device)
            # Compute prediction and loss
            logits = model(imgs['pixel_values'], texts)
        else:
            # Compute prediction and loss
            logits = model(imgs, texts)
        loss = loss_fn(logits, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(torch.argmax(logits, axis=1) == labels) / logits.shape[0]

        loss, current = loss.item(), (batch + 1) * batch_size
        if train_acc == 1:
            if loss < best_loss:
                torch.save(model.state_dict(), '/scratch1/yuqiuwan/CSCI567/trainedModels/bert_lunit_model_state_dict_last_epoch_output.pt')
            
      
        print(f"loss: {loss:>7f}; train_acc: {train_acc:>7f}  [{current:>5d}/{size:>5d}]")
            
######################## Train CLIP ########################
batch_size = 2
dropout = 0.0
learning_rate = 10**(-3)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# load phikon
image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
img_encoder = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
img_encoder.eval()

# load bert
text_encoder = BertModel.from_pretrained('bert-base-uncased')
text_encoder.eval()

clip_model = CLIP(img_encoder, text_encoder, d_embed=[384, 768], n_classes=5).to(device)

# Freeze the pre-trained model
for p in clip_model.image_encoder.parameters():
    p.requires_grad = False
for p in clip_model.text_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(clip_model.parameters(), lr=learning_rate)


image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
img_encoder = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
img_encoder.eval()
# load bert
text_encoder = BertModel.from_pretrained('bert-base-uncased')
text_encoder.eval()

clip_model = CLIP(img_encoder, text_encoder, d_embed=[384, 768], n_classes=5).to(device)
clip_model.load_state_dict(torch.load('/scratch1/yuqiuwan/CSCI567/trainedModels/bert_lunit_model_mask_state_dict_last_epoch_output.pt'))

class_ind = {'CC':0, 'EC':1, 'LGSC':2, 'HGSC':3, 'MC':4}

from torchmetrics.classification import MulticlassAUROC

def test_loop(dataloader, model, image_processor=None):
    size = len(dataloader.dataset)
    model.eval()
    texts = ['This type of cells have cell cytoplasm that are see through, and often have clear cell boundaries', 
             'Cells exhibit a back-to-back glandular pattern', 
             'This type of cells have cells close to normal healthy cells, cells containing single nuclei, and alive cell', 
             'There are many cells that are often deformed in shape, and many cells with multiple nucleus, and tissues often present many dead cells',
             'This type of cells often have goblet cells, they are often goblet-like or cell-like']

    texts = tokenizer(texts, padding='max_length', max_length=80, return_tensors='pt').to(device)
    
    correct_num = 0

    preds_temp = []
    labels_temp = []
    with torch.no_grad(): 
        for batch, (imgs, _, labels) in enumerate(dataloader):
            imgs = imgs.view(-1, 3, 224, 224)
            
            if image_processor:
                imgs = image_processor(imgs, return_tensors="pt").to(device)
                # Compute prediction and loss
                logits = model(imgs['pixel_values'], texts)
            else:
                # Compute prediction and loss
                logits = model(imgs, texts)
            preds_temp.append(logits)
            labels_temp.append(labels)

            correct_num += torch.sum(torch.argmax(logits, axis=1) == labels)
            print(batch, 'correct_num:', correct_num)

    mypreds = torch.cat(preds_temp, dim=0)
    mylabels = torch.cat(labels_temp, dim=0)
    metric = MulticlassAUROC(num_classes=5, average=None, thresholds=None)
    print("AUC: ", metric(mypreds, mylabels))
    # precision, recall, and f1
    precision = torchmetrics.Precision(task="multiclass", num_classes=5, average="none").to(device)
    recall = torchmetrics.Recall(task="multiclass", num_classes=5, average="none").to(device)
    precision_score = precision(mypreds, mylabels)
    recall_score = recall(mypreds, mylabels)
    test_acc = correct_num  / size

    acc = torchmetrics.Accuracy(task="multiclass", num_classes=5, average="none").to(device)
    accuracy = acc(mypreds, mylabels)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision_score)
    print("Recall: ", recall_score)
    print("accuracy: ", test_acc)

test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)
test_loop(test_dataloader, clip_model, image_processor)
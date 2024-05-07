import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAUROC
import torchmetrics

from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from ctran import ctranspath

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
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
            self.texts[c] = pd.read_table(text_dir+c+'.txt', header=None).iloc[:,0].to_list()
       
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
text_dir = "/scratch1/yuqiuwan/CSCI567/textLabelShortFilterted/"

wholedataset = OvarianDataset(metadata, image_dir, text_dir)
train_set, test_set = torch.utils.data.random_split(wholedataset, [0.8, 0.2])


class CLIP(nn.Module):
    def __init__(self, ImageEncoder, TextEncoder, d_embed=768, n_classes=5):
        super().__init__()
        self.image_encoder = ImageEncoder
        self.text_encoder = TextEncoder
        self.image_proj = nn.Linear(d_embed, n_classes)
        self.text_proj = nn.Linear(d_embed, n_classes)
        self.n_classes = n_classes

    def forward(self, img, text):
        img_outputs = self.image_encoder(img)
        img_features = img_outputs.cuda()
        img_embed = self.image_proj(img_features).view(-1, 100, self.n_classes)
        img_embed = torch.mean(img_embed, 1)
        img_embed = img_embed / torch.norm(img_embed, dim=-1, keepdim=True)

        text_outputs = self.text_encoder(**text)
        text_embed = text_outputs.pooler_output
        text_embed = self.text_proj(text_embed)
        text_embed = text_embed / torch.norm(text_embed, dim=-1, keepdim=True)

        logits = img_embed @ text_embed.T
        return logits
    

seq_max_length = 30
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

def contrastive_loss(logits, labels):
    image_loss = F.cross_entropy(logits, labels, reduction="mean")
    text_loss = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
    loss = (image_loss + text_loss) / 2

    return loss

def test_loop(dataloader, model, image_processor=None):
    size = len(dataloader.dataset)
    model.eval()
    # texts = ['Clear cell carcinoma is noted for its clear cytoplasm due to high glycogen content, often seen in cells that form distinctive patterns such as tubulocystic and papillary structures. It\'s appearance involves cells projecting into cystic spaces, contributing to the tumor\'s distinctive morphology.', 
    #            'Endometrioid carcinoma often mirrors the cellular structure of the endometrium, with well-formed glands and occasional squamous cells, presenting a histological diversity from well to poorly differentiated forms. It can present with various architectural patterns, including solid, cystic, and villoglandular, frequently associated with areas of hemorrhage and necrosis.', 
    #            'LGSC is less aggressive and features slow-growing cells with relatively uniform nuclei and minimal atypia, organized into intricate papillary structures. Despite its lower mitotic rate, this subtype can still exhibit psammoma bodies, though they are less common than in high-grade serous carcinoma.', 
    #            'This subtype is distinguished by its aggressive growth pattern and high mitotic index, featuring densely packed cells with irregular nuclei and prominent nucleoli. HGSC often shows extensive papillary and solid architecture, and may include psammoma bodies which are calcified deposits within the tumor.',
    #            'Mucinous carcinoma is characterized by its production of mucus, with tumors typically showing large cysts lined by tall columnar epithelial cells with abundant intracellular mucin. The mucin-filled cells often resemble those found in the gastrointestinal tract, making the tumors bulky and multilocular with a mixture of cystic and solid areas.']
    texts = ['have cytoplasm that see through, often have clear boundaries', 
               'exhibit a back-to-back glandular pattern', 
               'contain single nuclei, alive, but abnormal',
               'deformed in shape, tissues often present many dead',
               'these often exhibit goblet cells, usually taking on a goblet-like or cell-like appearance.']
    # texts = ['This type of cells have cell cytoplasm that are see through, and often have clear cell boundaries', 
    #          'Cells exhibit a back-to-back glandular pattern', 
    #          'This type of cells have cells close to normal healthy cells, cells containing single nuclei, and alive cell', 
    #          'There are many cells that are often deformed in shape, and many cells with multiple nucleus, and tissues often present many dead cells',
    #          'This type of cells often have goblet cells, they are often goblet-like or cell-like']
    texts = tokenizer(texts, padding='max_length', max_length=30, return_tensors='pt').to(device)
    
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
            print(logits, 'True label:', labels)

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

test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
img_encoder = ctranspath()
img_encoder.head = nn.Identity()
td = torch.load(r'./ctranspath.pth')
img_encoder.load_state_dict(td['model'], strict=True)
img_encoder.eval()

# load bert
text_encoder = BertModel.from_pretrained('bert-base-uncased')
text_encoder.eval()

clip_model = CLIP(img_encoder, text_encoder, d_embed=768, n_classes=5).to(device)
clip_model.load_state_dict(torch.load('/scratch1/yuqiuwan/CSCI567/ctranspath_clip_short_filtered.pt'))

test_loop(test_dataloader, clip_model, image_processor)
print("Done!")
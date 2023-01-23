import jcopdl
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from jcopdl.callback import Callback, set_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from jcopdl.utils.dataloader import MultilabelDataset
bs = 16
crop_size = 224

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(crop_size, scale = (0.9, 1.0)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor()
])

train_set = datasets.ImageFolder("C:/Users/abidq/OneDrive/Desktop/oil-spill-detection-main/train/", transform=train_transform)
trainloader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=1)

test_set = datasets.ImageFolder("C:/Users/abidq/OneDrive/Desktop/oil-spill-detection-main/test/", transform=test_transform)
testloader = DataLoader(test_set, batch_size=bs, shuffle=True)
label2cat = train_set.classes
label2cat
from torchvision.models import mobilenet_v2
mnet = mobilenet_v2(pretrained = True)
for param in mnet.parameters():
    param.requires_grad = False
mnet

mnet.classifier = nn.Sequential(
    nn.Linear(1280, 5),
    nn.LogSoftmax()
)
class CustomMobilenetV2(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=True)
        self.freeze()
        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, output_size),
            nn.LogSoftmax()
        )
        
    def forward(self,x):
        return self.mnet(x)
    
    def freeze(self):
        for param in mnet.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in mnet.parameters():
            param.requires_grad = True

config = set_config({
    "output_size" : len(train_set.classes),
    "batch_size" : bs,
    "crop_size" : crop_size
})
model = CustomMobilenetV2(config.output_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, early_stop_patience = 5, outdir="model")
from tqdm.auto import tqdm

def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc


   # fig.savefig("oilspill.jpg")
from PIL import Image
def image_loader(image_name):
    image = Image.open(image_name)
    image = test_transform(image).float()
    image = image.unsqueeze(0)  
    return image

def imagetaketaker(imagesource):
    images = image_loader(imagesource)
    images.shape
    model.load_state_dict(torch.load('model/weights_best.pth'))
    with torch.no_grad():
        model.eval()
        output = model (images)
        return(f"there is %s with probability %s" %(label2cat[output.argmax(1).item()], "{:.2f}".format(max(np.exp(output)[0]))))
        
#print(imagetaketaker("C:/Users/abidq/OneDrive/Desktop/oil-spill-detection-main/new_image/94123130_159289018877563_3786325641721819244_n.jpg"))



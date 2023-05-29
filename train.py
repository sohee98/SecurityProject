'''
Notes
Signature images are 3550
Celeb-A images are 202,599
'''
import os
import numpy as np
import math, sys
import glob, itertools
import argparse, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('tkagg')

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pdb

from models.unet import UNet

import warnings
warnings.filterwarnings("ignore")


seed = 12345

random.seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
# Set the random seed for PyTorch
torch.manual_seed(seed)
# Set the random seed for CUDA (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# load pretrained models
load_pretrained_models = False
# number of epochs of training
n_epochs = 100
# size of the batches
# batch_size = 512            # default
batch_size = 128
# name of the dataset
dataset_celeb = "./celeb_a/img_align_celeba"
dataset_sign = "./Dataset_All"
# adam: learning rate
lr = 0.00001
# adam: decay of first order momentum of gradient
b1 = 0.5
# adam: decay of first order momentum of gradient
b2 = 0.999
# number of cpu threads to use during batch generation
n_cpu = 20
# dimensionality of the latent space
latent_dim = 100
# size of each image dimension
img_size = 128
# size of random mask
mask_size = 64
# number of image channels
channels = 3
# interval between image sampling
sample_interval = 500

cuda = True if torch.cuda.is_available() else False
os.makedirs("saved_models", exist_ok=True)


class ImageDataset(Dataset):
    def __init__(self, celeb_root,sign_root, celeb_transforms_=None,sign_transforms_=None, img_size=128, sign_size=128, mode="train"):

        '''
        img = Celeb face image  202,599
        sign = signature        3550
        '''

        self.celeb_transform = transforms.Compose(celeb_transforms_)
        self.sign_transform = transforms.Compose(sign_transforms_)

        self.img_size = img_size
        self.sign_size = sign_size
        self.mode = mode

        self.celeb_files = glob.glob("%s/*.jpg" % celeb_root)
        self.sign_files = glob.glob("%s/*.png" % sign_root)
        random.shuffle(self.sign_files)

        #Pick random files exactly same size with sign_files
        self.celeb_files = random.sample(self.celeb_files,len(self.sign_files))     #3550

        #We split the dataset with the portion 7:3 (train : val)
        split_len = int(len(self.sign_files)*0.7)

        self.celeb_files = self.celeb_files[:split_len] if mode == "train" else self.celeb_files[split_len:]
        self.sign_files = self.sign_files[:split_len] if mode == "train" else self.sign_files[split_len:]

    def __getitem__(self, index):

        celeb_img = Image.open(self.celeb_files[index])
        sign_img = Image.open(self.sign_files[index])

        celeb_img = self.celeb_transform(celeb_img)
        sign_img = self.sign_transform(sign_img)

        return celeb_img, sign_img

    def __len__(self):
        return len(self.celeb_files)


celeb_transforms_ = [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

sign_transforms_ = [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]

test_celeb_transforms_ = [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.ToTensor(),
]
test_sign_transforms_ = [
    transforms.Resize((img_size, img_size), Image.BICUBIC),
    transforms.Grayscale(),
    transforms.ToTensor(),
]

train_dataloader = DataLoader(
    ImageDataset(dataset_celeb,dataset_sign, celeb_transforms_=celeb_transforms_ , sign_transforms_=sign_transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)

test_dataloader = DataLoader(
    ImageDataset(dataset_celeb,dataset_sign, celeb_transforms_=test_celeb_transforms_, sign_transforms_=test_sign_transforms_, mode="val"),
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_cpu,
)

# Loss function
MSE = nn.MSELoss()
TRIPLET = nn.TripletMarginWithDistanceLoss()

# Initialize generator and discriminator
unet = UNet(n_channels=channels,n_classes=3)
unet= torch.nn.DataParallel(unet)
unet2 = UNet(n_channels=channels,n_classes=4)
unet2= torch.nn.DataParallel(unet2)

if cuda:
    unet.cuda()
    unet2.cuda()
    MSE.cuda()

# Optimizers
unet_optimizer = torch.optim.Adam(unet.parameters(), lr=lr, betas=(b1, b2))
unet2_optimizer = torch.optim.Adam(unet2.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(n_epochs):

    ### Training ###
    tqdm_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch} ', total=int(len(train_dataloader)))

    for i, (celeb_image, sign_image) in enumerate(tqdm_bar):

        ### Stage 1
        unet_optimizer.zero_grad()
        unet.train()

        # Configure input
        celeb_image = celeb_image.type(Tensor)                 
        sign_image = sign_image.type(Tensor)      

        blended_image = celeb_image * 0.3 + sign_image * 0.7

        # Model using
        out_image = unet(blended_image)
        mse_loss = MSE(out_image, celeb_image)

        train_stage1_loss = mse_loss


        train_stage1_loss.backward()
        unet_optimizer.step()


        ### Stage 2
        unet2_optimizer.zero_grad()
        unet2.train()
        unet.eval()

        combined_image = unet2(out_image.clone().detach())

        restruct_celeb_image = combined_image[:,:-1]
        restruct_sign_image = combined_image[:,-1]
        restruct_sign_image = restruct_sign_image.unsqueeze(dim=1)

        expanded_sign_image = sign_image.expand_as(celeb_image)

        train_stage2_loss = TRIPLET(restruct_celeb_image,celeb_image,expanded_sign_image) *0.5 + MSE(restruct_sign_image,sign_image) *0.5

        train_stage2_loss.backward()
        unet2_optimizer.step()

    print('Train Loss : ', train_stage1_loss.item(), ' // ', train_stage2_loss.item())


    ### Evaluation ###
    with torch.no_grad():
        tqdm_bar = tqdm(test_dataloader, desc=f'Evaluation Epoch {epoch} ', total=int(len(test_dataloader)))

        for i, (celeb_image, sign_image) in enumerate(tqdm_bar):
            unet.eval()
            unet2.eval()

            ## Stage 1
            # Configure input
            celeb_image = celeb_image.type(Tensor)                 
            sign_image = sign_image.type(Tensor)      

            blended_image = celeb_image * 0.3 + sign_image * 0.7

            # Model using
            out_image = unet(blended_image)

            mse_loss = MSE(out_image, celeb_image)

            test_stage1_loss = mse_loss.item()

            ## Stage 2
            combined_image = unet2(out_image.clone().detach())

            restruct_celeb_image = combined_image[:,:-1]
            restruct_sign_image = combined_image[:,-1]
            restruct_sign_image = restruct_sign_image.unsqueeze(dim=1)

            expanded_sign_image = sign_image.expand_as(celeb_image)

            test_stage2_loss = TRIPLET(restruct_celeb_image,celeb_image,expanded_sign_image) *0.5 + MSE(restruct_sign_image,sign_image) *0.5

        print('Evaluation Loss : ', test_stage1_loss, ' // ', test_stage2_loss.item())
      
torch.save(unet.state_dict(), f"saved_models/unet.pth")
torch.save(unet2.state_dict(), f"saved_models/unet2.pth")

print('Enjoy')

pdb.set_trace()
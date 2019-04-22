import os 
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from model_utils import load_data

train_path = '/nethome/zyu336/dl_catdog/data/train/'

img_shape = (3, 224, 224)
batch_size = 16
n_epochs = 20
sample_interval = 400

cuda = torch.cuda.is_available()
latent_dim = 256
train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
dataloader, _, n_classes, _, _, _, _ = load_data('breeds_cat', train_transform, train_transform, batch_size)

n_classes = len(n_classes)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

class Unflatten(nn.Module):
    def __init__(self, N=-1, C=3, H=224, W=224):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.COnvTranspose2d):
        init.xavier_uniform(m.weight.data)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(128, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 3 * 224 * 224),
                nn.Tanh()
       ) 
    def forward(self, noise):
        x = self.model(noise)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                Flatten(),
                nn.Linear(3*224*224, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 64),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

def sample_image(n_row, batches_done):
    noise = sample_noise(1, noise_size)
    gen_imgs = generator(Variable(noise.cuda()))
    gen_imgs = gen_imgs.view(3, 224, 224)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

Bce_loss = nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real, logits_fake):
    loss = None
    N = logits_real.size()
    true_labels = Variable(torch.ones(N).cuda())
    real_image_loss = Bce_loss(logits_real, true_labels)
    fake_image_loss = Bce_loss(logits_fake, 1 - true_labels)

    loss = real_image_loss + fake_image_loss

    return loss

def generator_loss(logits_fake):
    N = logits_fake.size()
    true_labels = Variable(torch.ones(N).cuda())
    loss = Bce_loss(logits_fake, true_labels)

    return loss


d_loss_list = []
g_loss_list = []
noise_size = 128

def sample_noise(batch_size, dim):
    temp = torch.rand(batch_size, dim) + torch.rand(batch_size, dim)*(-1)
    return temp


for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
        optimizer_D.zero_grad()

        real_data = Variable(imgs.cuda())
        logits_real = discriminator(2* (real_data - 0.5))
        g_fake_seed = Variable(sample_noise(batch_size, noise_size).cuda())
        fake_images = generator(g_fake_seed).detach()
        logits_fake = discriminator(fake_images.view(batch_size, 3, 224, 224))

        d_loss = discriminator_loss(logits_real, logits_fake)
        d_loss.backward()

        optimizer_G.zero_grad()

        g_fake_seed = Variable(sample_noise(batch_size, noise_size).cuda())
        fake_images = generator(g_fake_seed)

        gen_logits_fake = discriminator(fake_images.view(batch_size, 3, 224, 224))
        g_loss = generator_loss(gen_logits_fake)
        g_loss.backward()

        optimizer_G.step()

        print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
             )

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            sample_image(n_row=1, batches_done=batches_done)


indices = list(range(0, len(d_loss_list)))

import matplotlib.pyplot as plt

plt.plot(indices, d_loss_list)
plt.xlabel('Batches')
plt.ylabel('D Loss')
plt.title('Discriminator Loss')

plt.savefig('d_loss_lc.png')
plt.clf()

plt.plot(indices, g_loss_list)
plt.xlabel('Batches')
plt.ylabel('G Loss')
plt.title('Generator Loss')
plt.savefig('g_loss_lc.png')



























































import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Discriminator, Generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=28, help="size of image height")
    parser.add_argument("--img_width", type=int, default=28, help="size of image height")
    parser.add_argument("--z_dim", type=int, default=100, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
    )
    return parser.parse_args()


def main(cfg):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.MNIST(root='../mnist_data/', train=True, transform=transform, download=True)
    # test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

    # Data Loader
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.n_cpu,
        shuffle=True
    )
    
    # imgage size
    img_shape = (cfg.channels, cfg.img_width, cfg.img_height)
    img_dim = cfg.img_width * cfg.img_height 

    # build network
    generator = Generator(input_dim=cfg.z_dim, output_dim=img_dim).to(device)
    discriminator = Discriminator(img_dim).to(device)

    # loss
    adversarial_loss = nn.BCELoss().to(device)

    # optimizer
    G_optimizer = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    
    train(cfg, generator, discriminator, G_optimizer, D_optimizer, train_loader, adversarial_loss, device)

        
def train(cfg, generator, discriminator, G_optimizer, D_optimizer, train_loader, criterion, device):
            
    for epoch in range(cfg.epoch, cfg.n_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            
            # Configure input
            real_imgs = imgs.view(imgs.size(0), -1).to(device)
        
            # Adversarial ground truths
            valid = torch.ones(real_imgs.size(0), 1).to(device)
            fake = torch.zeros(real_imgs.size(0), 1).to(device)
            
            # ------------------
            #  Train Generators
            # ------------------
            G_optimizer.zero_grad()
            
            z = torch.randn(real_imgs.size(0), cfg.z_dim).to(device)
            
            # Generator loss
            fake_G = generator(z)
            pred_fake = discriminator(fake_G)
            G_loss = criterion(pred_fake, valid)

            G_loss.backward()
            G_optimizer.step()
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            D_optimizer.zero_grad()

            # Dicriminator Loss
            
            # real Loss
            pred_real = discriminator(real_imgs)
            D_real_loss = criterion(pred_real, valid)

            # fake Loss
            z = torch.randn(real_imgs.size(0), cfg.z_dim).to(device)
            fake_G = generator(z)
            
            pred_fake = discriminator(fake_G)
            D_fake_loss = criterion(pred_fake, fake)

            # Total Loss
            D_loss = D_real_loss + D_fake_loss
            
            D_loss.backward()
            D_optimizer.step()

            if i % 100 == 0:
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                    % (epoch, cfg.n_epochs, i, len(train_loader), D_loss.item(), G_loss.item())
                )
            
            
        # If at sample interval save image
        if epoch % cfg.sample_interval == 0:
            with torch.no_grad():
                test_z = torch.randn(100, cfg.z_dim).to(device)
                generated = generator(test_z)

                save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + str(epoch) + '.png', nrow=10)
        
if __name__ == '__main__':
    cfg = parse_args()
    os.makedirs("samples", exist_ok=True)
    main(cfg)

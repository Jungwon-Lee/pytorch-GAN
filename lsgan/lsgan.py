import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=32, help="size of image height")
    parser.add_argument("--img_width", type=int, default=32, help="size of image height")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--z_dim", type=int, default=100, help="size of image width")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
    )
    return parser.parse_args()

def main(cfg):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.Resize(cfg.img_width),
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
    generator = Generator(input_dim=cfg.z_dim, img_size=cfg.img_width, channels=cfg.channels).to(device)
    discriminator = Discriminator(img_size=cfg.img_width, channels=cfg.channels).to(device)
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # least squares loss
    adversarial_loss = nn.MSELoss().to(device)
    
    # optimizer
    G_optimizer = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    train(cfg, generator, discriminator, G_optimizer, D_optimizer, train_loader, adversarial_loss, device)


def train(cfg, generator, discriminator, G_optimizer, D_optimizer, dataloader, adversarial_loss, device):
    
    for epoch in range(cfg.epoch, cfg.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            
            # Configure input
            real_imgs = imgs.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(real_imgs.size(0), 1).to(device)
            fake = torch.zeros(real_imgs.size(0), 1).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            G_optimizer.zero_grad()

            z = torch.normal(mean=torch.zeros((real_imgs.size(0), cfg.z_dim)), std=torch.ones((real_imgs.size(0), cfg.z_dim))).to(device)
            
            # Generator loss
            fake_G = generator(z)
            
            pred_fake = discriminator(fake_G)
            G_loss = adversarial_loss(pred_fake, valid)

            G_loss.backward()
            G_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            D_optimizer.zero_grad()

            # Dicriminator Loss
            
            # generate fake image
            z = torch.normal(mean=torch.zeros((real_imgs.size(0), cfg.z_dim)), std=torch.ones((real_imgs.size(0), cfg.z_dim))).to(device)
            fake_G = generator(z)
            
            # real loss
            pred_real = discriminator(real_imgs)
            D_real_loss = adversarial_loss(pred_real, valid)

            # fake loss
            pred_fake = discriminator(fake_G)
            D_fake_loss = adversarial_loss(pred_fake, fake)

            # Total loss
            D_loss = 0.5 * (D_real_loss + D_fake_loss)

            D_loss.backward()
            D_optimizer.step()

            if i % 100 == 0:
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                    % (epoch, cfg.n_epochs, i, len(dataloader), D_loss.item(), G_loss.item())
                )

        if epoch % cfg.sample_interval == 0:
            with torch.no_grad():
                test_z = torch.randn(100, cfg.z_dim).to(device)
                generated = generator(test_z)

                save_image(generated, './samples/sample_' + str(epoch) + '.png', nrow=10)             
             
if __name__ == '__main__':
    cfg = parse_args()
    os.makedirs("samples", exist_ok=True)
    main(cfg)

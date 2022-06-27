import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from dataset import *
from model import Generator as Pix2Pix
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image height")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--lambda_pixel", type=int, default=100, help="lambda_pixel")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
    )
    return parser.parse_args()

def main(cfg):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((cfg.img_width, cfg.img_height), transforms.InterpolationMode.BICUBIC),
    ])

    # Data Loader
    train_loader = DataLoader(
        dataset=ImageDataset('maps', transforms_=transform, train=True),
        batch_size=cfg.batch_size, 
        num_workers=cfg.n_cpu,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=ImageDataset('maps', transforms_=transform, train=False),
        batch_size=2, 
        num_workers=1,
        shuffle=False
    )
    
    # test sample image
    test_real_A, test_real_B = next(iter(val_loader))
    test_real_A = test_real_A.to(device)
    test_real_B = test_real_B.to(device)

    # patch
    patch = (1, cfg.img_height // 2 ** 4, cfg.img_width // 2 ** 4)

    # build network
    generator = Pix2Pix(in_channels=cfg.channels, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=2*cfg.channels).to(device)
    
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss functions
    adversarial_loss = nn.MSELoss().to(device)
    pixelwise_loss = nn.L1Loss().to(device)
    
    # optimizer
    G_optimizer = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

    train(cfg, generator, discriminator, G_optimizer, D_optimizer, train_loader, adversarial_loss, pixelwise_loss, patch, test_real_A, test_real_B, device)


def train(cfg, generator, discriminator, G_optimizer, D_optimizer, dataloader, adversarial_loss, pixelwise_loss, patch, test_real_A, test_real_B, device):
    
    for epoch in range(cfg.epoch, cfg.n_epochs):
        for i, (img1, img2) in enumerate(dataloader):
            
            # Configure input
            real_A = img1.to(device)
            real_B = img2.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(real_A.size(0), *patch).to(device)
            fake = torch.zeros(real_A.size(0), *patch).to(device)

            # -----------------
            #  Train Generator
            # -----------------
            G_optimizer.zero_grad()

            # Generator loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            
            adv_loss = adversarial_loss(pred_fake, valid)
            loss_pixel = pixelwise_loss(fake_B, real_B)
            
            G_loss = adv_loss + cfg.lambda_pixel * loss_pixel
            
            G_loss.backward()
            G_optimizer.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            D_optimizer.zero_grad()

            # Dicriminator Loss
            
             # real Loss
            pred_real = discriminator(real_B, real_A)
            D_real_loss = adversarial_loss(pred_real, valid)

            # fake Loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            D_fake_loss = adversarial_loss(pred_fake, fake)

            # Total loss
            D_loss = 0.5 * (D_real_loss + D_fake_loss)

            D_loss.backward()
            D_optimizer.step()

            if i % 10 == 0:
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                    % (epoch, cfg.n_epochs, i, len(dataloader), D_loss.item(), G_loss.item())
                )

        if epoch % cfg.sample_interval == 0:
            with torch.no_grad():
                generated = generator(test_real_A)
                img_sample = torch.cat((test_real_A.data, generated.data, test_real_B.data), -2)
                save_image(img_sample, './samples/sample_' + str(epoch) + '.png', nrow=2, normalize=True)            
             
if __name__ == '__main__':
    cfg = parse_args()
    os.makedirs("samples", exist_ok=True)
    main(cfg)

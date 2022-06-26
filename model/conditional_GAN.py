import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, n_classes):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.output_dim = output_dim
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.output_dim)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(input_dim)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class GAN(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = cfg.latent_dim
        self.lr = cfg.lr
        self.b1 = cfg.b1
        self.b2 = cfg.b2
        self.batch_size = cfg.batch_size
        self.n_classes = cfg.n_classes
        
        self.img_shape = cfg.mnist_shape
        self.img_dim = np.prod(cfg.mnist_shape) # (1, 28, 28)
        
        # networks
        self.generator = Generator(input_dim=self.latent_dim, output_dim=self.img_dim, n_classes=self.n_classes)
        self.discriminator = Discriminator(self.img_dim, n_classes=self.n_classes)
  
        self.validation_z = torch.randn(8, self.latent_dim)
        self.validation_label = torch.randint(0, self.n_classes, (8,))
        
        self.example_input_array = (
            torch.zeros(2, self.latent_dim),
            torch.LongTensor(np.random.randint(0, self.n_classes, 2)))

    def forward(self, z, label):
        return self.generator(z, label)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        print(batch_idx, flush=True)
        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)
        
        # fake label
        gen_labels = torch.randint(0, self.n_classes, (imgs.shape[0],))
        gen_labels = gen_labels.type_as(labels)
        
        # generate fake images
        fake_image = self(z, gen_labels)
            
        # train generator
        if optimizer_idx == 0:
            
            # log sampled images
            sample_imgs = fake_image[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            pred_fake = self.discriminator(fake_image, gen_labels)
            g_loss = self.adversarial_loss(pred_fake, valid)
        
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            # real loss
            pred_real = self.discriminator(imgs, labels)
            real_loss = self.adversarial_loss(pred_real, valid)

            # fake loss
            pred_fake = self.discriminator(fake_image.detach(), gen_labels)
            fake_loss = self.adversarial_loss(pred_fake, fake)

            # discriminator loss is the average of these
            d_loss = 0.5 * (real_loss + fake_loss)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)
        label = self.validation_label.to(self.device)
        
        # log sampled images
        sample_imgs = self(z, label)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


"""
pytorch-lighting gan template
python train.py --gan_model
"""

from argparse import ArgumentParser
from importlib import import_module

from pytorch_lightning.trainer import Trainer

from Config import Config
from model.GAN import GAN


def main(cfg):

    # ------------------------
    # 0 import GAN model
    # ------------------------
    # It means <from GAN import GAN as GAN>
    # module = import_module('GAN.lightning.model.' + cfg.model_name)
    # GAN = getattr(module, cfg.model_name)
    
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = GAN(cfg)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    trainer = Trainer(gpus=cfg.gpus)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, help="GAN model name")
    opt = parser.parse_args()
    
    cfg = Config.from_yaml('config/' + opt.model_name + '.yml')
    main(cfg)

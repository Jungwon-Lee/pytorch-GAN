import glob

from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class ImageDataset(Dataset):
    def __init__(self, dataset_name, transforms_=None, train=True):
        super(ImageDataset).__init__()
        
        if train:
            self.data_list = glob.glob('datasets/' + dataset_name + '/train/*.jpg')
        else:
            self.data_list = glob.glob('datasets/' + dataset_name + '/val/*.jpg')
        
        self.transforms_ = transforms_

    def __getitem__(self, index):
        imgs = read_image(self.data_list[index], ImageReadMode.RGB)
        
        real_A = imgs[:,:,imgs.size(-1)//2:]
        real_B = imgs[:,:,:imgs.size(-1)//2]
        
        if self.transforms_:
            real_A = real_A / 127.5 -1
            real_B = real_B / 127.5 -1
            real_A = self.transforms_(real_A) 
            real_B = self.transforms_(real_B)

        return real_A, real_B

    def __len__(self):
        return len(self.data_list)

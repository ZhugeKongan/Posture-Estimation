import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import math
NUM_CLASSES=71
class MegaAsiaAgeDataset(Dataset):

    def __init__(self, data_dir,label_file, transform=None):
        '''
        img_dir: 图片路径：img_dir + img_name.jpg构成图片的完整路径
        '''
        # 所有图片的绝对路径
        self.data_dir=data_dir
        self.imgs=np.loadtxt(label_file+'_name.txt',dtype='str')
        self.labels=np.loadtxt(label_file+'_age.txt')
        self.transform = transform
    def normal_sampling(self,mean, label_k, std=0.5):
        return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir,self.imgs[index]))#.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # print(float(label))
        label = self.labels[index]
        label = int(label)
        levels = [1] * label + [0] * (NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        # labels = [self.normal_sampling(int(label), i) for i in range(4)]
        # labels = [i if i > 1e-10 else 1e-10 for i in labels]
        # labels = torch.Tensor(labels)

        return img, label, levels

    def __len__(self):
        return len(self.imgs)



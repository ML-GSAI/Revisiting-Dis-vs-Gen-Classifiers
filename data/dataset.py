import torch
from torch.utils.data import Dataset
import os
import numpy as np

class SimDataset(Dataset):
    def __init__(self, root, features, K, train=True):
        self.features = features
        self.K = K
        self.imgs = []
        self.labels = []

        if train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        root_dir = os.listdir(self.root)
        label = 0
        for son_dir in root_dir:
            son_dir_imgs = os.listdir(os.path.join(self.root, son_dir))
            for i in range(len(son_dir_imgs)):
                self.imgs.append(os.path.join(self.root, son_dir, son_dir_imgs[i]).replace('\\', '/'))
            self.labels += len(son_dir_imgs) * [label]
            label += 1

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = np.load(img_path)
        img = torch.tensor(img, dtype=torch.float)
        if self.K > 2:
            img -= (2 ** (self.K - 3))
            img /= (2 ** (self.K - 3))
        label = self.labels[index]

        return img, label

if __name__ == '__main__':
    dataset = SimDataset(root='datasets/sim_normal/K5_n100', features=100, K=5)
    print(dataset[1000])
    print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # for img, labels in dataloader:
    #     print(img)
    #     print(labels)
    #     break

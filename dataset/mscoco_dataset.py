from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import torch


class mscocoDataset(Dataset):
    def __getitem__(self, index):
        item = self.file['train2014/images'][index]

        return self.transform(item) * 255

    def __len__(self):
        return self.file['train2014/images'].shape[0]

    def __init__(self, file='../data/ms-coco-256.h5'):
        self.file = h5py.File(file, mode='r')
        self.transform = transforms.Compose([
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1])
        ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = mscocoDataset()
    plt.figure(figsize=(12, 12))
    for i in range(len(dataset)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.transpose(dataset[i].numpy(), (1, 2, 0)))
    plt.show()

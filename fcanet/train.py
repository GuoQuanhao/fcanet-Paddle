from PIL import Image
import numpy as np
from paddle.io import Dataset, DataLoader
import scipy.io
import os

# A = TrainDataset('../Data/benchmark_RELEASE/dataset')
class TrainDataset(Dataset):
    def __init__(self,dataset_path, img_folder='img', gt_folder='cls',threshold=128,ignore_label=None):
        self.threshold, self.ignore_label = threshold, ignore_label
        dataset_path=dataset_path
        with open(os.path.join(dataset_path, 'train.txt'), 'r', encoding='utf-8') as file:
            img_files = file.readlines()
        img_files = [img.rstrip('\n') for img in img_files]
        gt_files = [os.path.join(dataset_path, gt_folder, img+'.mat') for img in img_files]
        img_files = [os.path.join(dataset_path, img_folder, img+'.jpg') for img in img_files]
        self.img_files = img_files
        self.gt_files = gt_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_src = np.array(Image.open(self.img_files[idx]))
        gt_src = scipy.io.loadmat(self.gt_files[idx])
        gt = gt_src['GTcls'][0][0][-2]
        return img_src, gt
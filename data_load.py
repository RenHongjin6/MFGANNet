import glob, os, torch
from PIL import Image
from utils import *
import numpy as np
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_of_img1, path_of_img2, path_of_label, path_of_csv, mode='train'):
        super().__init__()
        self.mode = mode
        self.img1_list = glob.glob(os.path.join(path_of_img1, '*.jpg'))# *指取全部文件
        self.img2_list = glob.glob(os.path.join(path_of_img2, '*.jpg'))
        self.label_list = glob.glob(os.path.join(path_of_label,'*.png'))
        self.label_info = get_label_info(path_of_csv)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img1 = Image.open(self.img1_list[index])
        img2 = Image.open(self.img2_list[index])
        label = Image.open(self.label_list[index])

        label = one_hot_it(label, self.label_info).astype(np.uint8)
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)

        img1 = self.to_tensor(img1).float()
        img2 = self.to_tensor(img2).float()

        return img1, img2, label

    def __len__(self):
        return len(self.img1_list)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = Dataset(
        "/data/train/img1",
        "/datasets/data/train/img2",
        "/datasets/data/train/label",
        "/datasets/data/class_dict.csv",
    )
    dataloader_test=DataLoader(
        data,
        batch_size=3,
        shuffle=True,
        num_workers=6,
    )

    for i, (img1, img2, label) in enumerate(dataloader_test):
        print(img1)
        print(img2)
        print(label)
        print(i)
        if i == 20:
            print("************测试结束**************")
            break

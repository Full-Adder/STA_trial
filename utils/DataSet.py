import os

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import utils.DataFromtxt as dft

Data_path = r"../AVE_Dataset"
Video_path = r"../AVE_Dataset/Video"
Pic_path = r"../AVE_Dataset/Picture"
Audio_path = r"../AVE_Dataset/Audio"
H5_path = r"../AVE_Dataset/H5"


class AVEDataset(Dataset):  # 数据集类
    def __init__(self, pic_dir, h5_dir, mode, transform=None, STA_mode="S"):
        assert mode in list(dft.get_txtList().keys()), "mode must be train/test/val"
        assert STA_mode in ["S", "ST", "SA", "STA"], "STA_mode must be S/SA/ST/STA"
        self.pic_dir = pic_dir
        self.h5_dir = h5_dir
        self.mode = mode
        self.transform = transform
        self.STA_mode = STA_mode
        self.class_dir = dft.get_category_to_key()
        data_folder_list = dft.readDataTxt(os.path.join(self.pic_dir, "../"), mode)
        self.data_list = []
        for idata in data_folder_list:
            for idx in range(idata[-2], idata[-1]):
                self.data_list.append([os.path.join(idata[0], "{:0>2d}".format(idx)), idata[1]])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        img_path = os.path.join(self.pic_dir, data[0]+".jpg")  # 图片绝对地址
        image = Image.open(img_path).convert('RGB')  # 打开图片，并转化为RGB格式
        image = self.transform(image)  # 将图片转化为tensor

        class_id = int(data[1])  # 获得图像标签
        label = np.zeros(len(self.class_dir), dtype=np.float32)
        label[class_id] = 1  # one-hot编码

        if self.STA_mode == "S":
            return data[0], image, class_id, label  # 返回 图片地址 图片tensor 标签 onehot标签

        elif self.STA_mode == "SA":
            h5_path = os.path.join(self.h5_dir, data[0]+".h5")
            with h5py.File(h5_path, 'r') as hf:
                audio_features = np.float32(hf['dataset'][:])  # 5,128
            audio = torch.from_numpy(audio_features).float()
            return data[0], image, audio, class_id, label  # 返回 图片地址 图片tensor 标签 onehot标签

        elif self.STA_mode == "ST":
            pass

        else:   # "STA"
            pass


if __name__ == "__main__":
    x = AVEDataset(pic_dir="../AVE_Dataset/Picture", h5_dir=r"../AVE_Dataset/H5",
                   mode="train", transform=transforms.ToTensor(), STA_mode="SA")
    print(len(x))
    for i in x:
        print(i)
        break

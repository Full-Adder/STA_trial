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
    def __init__(self, pic_dir, h5_dir, gt_dir, mode, transform, STA_mode):
        assert mode in list(dft.get_txtList().keys()), "mode must be train/test/val"
        assert STA_mode in ["S", "ST", "SA", "STA"], "STA_mode must be S/SA/ST/STA"
        self.pic_dir = pic_dir
        self.h5_dir = h5_dir
        self.gt_dir = gt_dir
        self.mode = mode
        self.transform = transform
        self.STA_mode = STA_mode
        self.class_dir = dft.get_category_to_key()
        data_folder_list = dft.readDataTxt(os.path.join(self.pic_dir, "../"), mode)
        self.data_list = []
        for idata in data_folder_list:
            if (self.STA_mode == "S" or self.STA_mode == "SA") and mode != "all":
                for idx in range(idata[-2], idata[-1]):
                    self.data_list.append([os.path.join(idata[0], "{:0>2d}".format(idx)), idata[1]])
            else:
                for idx in range(idata[-2] + 1, idata[-1] - 1):
                    self.data_list.append([os.path.join(idata[0], "{:0>2d}".format(idx)), idata[1]])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        video_name, id = data[0][:11], int(data[0][-2:])

        img_path = os.path.join(self.pic_dir, video_name, "%02d" % id + ".jpg")  # 图片绝对地址
        image = Image.open(img_path).convert('RGB')  # 打开图片，并转化为RGB格式
        image = self.transform(image)  # 将图片转化为tensor

        class_id = int(data[1])  # 获得图像标签
        onehot_label = np.zeros(len(self.class_dir), dtype=np.float32)
        onehot_label[class_id] = 1  # one-hot编码

        if self.STA_mode == "S":
            return data[0], image, class_id, onehot_label  # 返回 图片地址 图片tensor 标签 onehot标签

        h5_path = os.path.join(self.h5_dir, data[0] + ".h5")
        with h5py.File(h5_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio = torch.from_numpy(audio_features).float()

        if self.STA_mode == "SA":
            return data[0], image, audio, class_id, onehot_label  # 返回 图片地址 图片tensor 标签 onehot标签

        img_bef_path = os.path.join(self.pic_dir, video_name, "%02d" % (id - 1) + ".jpg")  # 前一张图片的地址
        img_aft_path = os.path.join(self.pic_dir, video_name, "%02d" % (id + 1) + ".jpg")  # 后一张图片的地址
        image_bef = Image.open(img_bef_path).convert('RGB')
        image_aft = Image.open(img_aft_path).convert('RGB')
        image_bef = self.transform(image_bef)
        image_aft = self.transform(image_aft)

        if self.STA_mode == "ST":
            return data[0], image_bef, image, image_aft, class_id, onehot_label

        aud_bef_path = os.path.join(self.h5_dir, video_name, "%02d" % (id - 1) + ".h5")
        aud_aft_path = os.path.join(self.h5_dir, video_name, "%02d" % (id + 1) + ".h5")
        with h5py.File(aud_bef_path, 'r') as hf:
            audio_features_bef = np.float32(hf['dataset'][:])  # 5,128
        with h5py.File(aud_aft_path, "r") as hf:
            audio_features_aft = np.float32(hf['dataset'][:])
        aud_bef = torch.from_numpy(audio_features_bef).float()
        aud_aft = torch.from_numpy(audio_features_aft).float()

        
        
        return data[0], image_bef, aud_bef, gt_bef, image, audio, gt_now, image_aft, aud_aft, gt_aft, class_id, onehot_label


if __name__ == "__main__":
    x = AVEDataset(pic_dir="../AVE_Dataset/Picture", h5_dir=r"../AVE_Dataset/H5", gt_dir="./",
                   mode="train", transform=transforms.ToTensor(), STA_mode="S")
    print(len(x))
    for i in x:
        print(i)
        break

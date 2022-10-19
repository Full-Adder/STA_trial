import os
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import utils.DataFromtxt as dft
from utils.args_config import get_parser


class AVEDataset(Dataset):  # 数据集类
    def __init__(self, pic_dir, h5_dir, gt_dir, STA_mode, mode, transform, transforms_gt):
        assert mode in dft.get_txtList().keys(), "mode must be train/test"
        assert STA_mode in ["S", "ST", "SA", "STA"], "STA_mode must be S/SA/ST/STA"
        self.pic_dir = pic_dir
        self.h5_dir = h5_dir
        self.gt_dir = gt_dir
        self.mode = mode
        self.transform = transform
        self.transform_gt = transforms_gt
        self.STA_mode = STA_mode
        self.class_dir = dft.get_category_to_key()
        data_folder_list = dft.readDataTxt(os.path.join(self.pic_dir, "../"), mode)
        self.data_list = []
        for idata in data_folder_list:
            if idata[-1] - idata[-2] <= 3:
                continue
            else:
                for idx in range(idata[-2], idata[-1]):
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

        if os.path.exists(img_bef_path):
            image_bef = Image.open(img_bef_path).convert('RGB')
            image_bef = self.transform(image_bef)
        else:
            image_bef = image

        if os.path.exists(img_aft_path):
            image_aft = Image.open(img_aft_path).convert('RGB')
            image_aft = self.transform(image_aft)
        else:
            image_aft = image

        if self.STA_mode == "ST":
            return data[0], image_bef, image, image_aft, class_id, onehot_label

        gt_now_path = os.path.join(self.gt_dir, video_name, "%02d" % id + ".jpg")
        gt_now = self.transform_gt(Image.open(gt_now_path).convert('L'))

        gt_bef_path = os.path.join(self.gt_dir, video_name, "%02d" % (id - 1) + ".jpg")
        gt_aft_path = os.path.join(self.gt_dir, video_name, "%02d" % (id + 1) + ".jpg")

        if os.path.exists(gt_bef_path):
            aud_bef_path = os.path.join(self.h5_dir, video_name, "%02d" % (id - 1) + ".h5")
            with h5py.File(aud_bef_path, 'r') as hf:
                audio_features_bef = np.float32(hf['dataset'][:])  # 5,128
            aud_bef = torch.from_numpy(audio_features_bef).float()
            gt_bef = self.transform_gt(Image.open(gt_bef_path).convert('L'))
        else:
            aud_bef = audio
            gt_bef = gt_now

        if os.path.exists(gt_aft_path):
            aud_aft_path = os.path.join(self.h5_dir, video_name, "%02d" % (id + 1) + ".h5")
            with h5py.File(aud_aft_path, "r") as hf:
                audio_features_aft = np.float32(hf['dataset'][:])
            aud_aft = torch.from_numpy(audio_features_aft).float()
            gt_aft = self.transform_gt(Image.open(gt_aft_path).convert('L'))
        else:
            aud_aft = audio
            gt_aft = gt_now

        return data[0], image_bef, aud_bef, gt_bef, image, audio, gt_now, image_aft, aud_aft, gt_aft, class_id, onehot_label


if __name__ == "__main__":
    args = get_parser()
    x = AVEDataset(pic_dir=args.Pic_path, h5_dir=args.H5_path, gt_dir=args.GT_path,
                   mode="train", transform=transforms.ToTensor(), transforms_gt=None, STA_mode="S")
    print(len(x))
    for i in x:
        print(i)
        break

import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
from PIL import Image


def get_filename_list_form_txt(txt_path):
    name_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for data in f.readlines():
            name, st, en = [i for i in data.strip().split('&')]
            name_list.append([name, int(st), int(en)])
    return name_list


class OtherDataset(Dataset):  # 数据集类
    def __init__(self, pic_dir, h5_dir, transform):
        self.pic_dir = pic_dir
        self.h5_dir = h5_dir
        self.transform = transform
        data_folder_list = get_filename_list_form_txt(os.path.join(pic_dir, r"../filename.txt"))
        self.data_list = []
        for idata in data_folder_list:
            if idata[-1] - idata[-2] <= 3:
                continue
            else:
                for idx in range(idata[-2]+1, idata[-1]-1):
                    self.data_list.append([os.path.join(idata[0], "{:0>4d}".format(idx))])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        # print(data)
        video_name, id = data[0][:-4], int(data[0][-4:])

        img_bef_path = os.path.join(self.pic_dir, video_name, "%04d" % (id - 1) + ".jpg")  # 前一张图片的地址
        img_path = os.path.join(self.pic_dir, video_name, "%04d" % id + ".jpg")  # 图片绝对地址
        img_aft_path = os.path.join(self.pic_dir, video_name, "%04d" % (id + 1) + ".jpg")  # 后一张图片的地址

        image = Image.open(img_path).convert('RGB')  # 打开图片，并转化为RGB格式
        image = self.transform(image)  # 将图片转化为tensor

        h5_path = os.path.join(self.h5_dir, data[0] + ".h5")
        with h5py.File(h5_path, 'r') as hf:
            audio_features = np.float32(hf['dataset'][:])  # 5,128
        audio = torch.from_numpy(audio_features).float()

        aud_bef_path = os.path.join(self.h5_dir, video_name, "%04d" % (id - 1) + ".h5")
        if os.path.exists(aud_bef_path):
            image_bef = Image.open(img_bef_path).convert('RGB')
            image_bef = self.transform(image_bef)

            with h5py.File(aud_bef_path, 'r') as hf:
                audio_features_bef = np.float32(hf['dataset'][:])  # 5,128
            aud_bef = torch.from_numpy(audio_features_bef).float()
        else:
            image_bef = image
            aud_bef = audio

        aud_aft_path = os.path.join(self.h5_dir, video_name, "%04d" % (id + 1) + ".h5")
        if os.path.exists(aud_aft_path):
            image_aft = Image.open(img_aft_path).convert('RGB')
            image_aft = self.transform(image_aft)

            with h5py.File(aud_aft_path, "r") as hf:
                audio_features_aft = np.float32(hf['dataset'][:])
            aud_aft = torch.from_numpy(audio_features_aft).float()
        else:
            image_aft = image
            aud_aft = audio

        return data[0], image_bef, aud_bef, image, audio, image_aft, aud_aft


def get_otherDataLoader(Pic_path, H5_path, batch_size, input_size):
    mean_vals = [0.485, 0.456, 0.406]  # 数据均值
    std_vals = [0.229, 0.224, 0.225]  # 数据标准差

    tsfm_img = transforms.Compose([transforms.Resize((input_size, input_size)),
                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean_vals, std_vals),
                                   ])
    dataset = OtherDataset(pic_dir=Pic_path, h5_dir=H5_path, transform=tsfm_img)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    print("TrainSet.len:", len(dataset), "\t dataLoader.len:", len(dataloader), 'batch_size:', batch_size)
    return dataloader

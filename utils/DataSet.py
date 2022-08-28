import os

from torch.utils.data import DataLoader, Dataset
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
    def __init__(self, data_dir, mode, transform=None):
        assert mode in list(dft.get_txtList().keys()), "mode must be train/test/val"
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.class_dir = dft.get_category_to_key()
        data_folder_list = dft.readDataTxt(os.path.join(data_dir, "../"), mode)
        self.data_list = []
        for idata in data_folder_list:
            for idx in range(idata[-2], idata[-1]):
                self.data_list.append([os.path.join(data_dir, idata[0], "{:0>2d}.jpg".format(idx)), idata[1]])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        img_path = data[0]  # 图片绝对地址
        image = Image.open(img_path).convert('RGB')  # 打开图片，并转化为RGB格式
        image = self.transform(image)  # 将图片转化为tensor

        class_id = int(data[1])  # 获得图像标签
        label = np.zeros(len(self.class_dir), dtype=np.float32)
        label[class_id] = 1  # one-hot编码
        return img_path, image, class_id, label  # 返回 图片地址 图片tensor 标签 onehot标签


if __name__ == "__main__":
    x = AVEDataset("../AVE_Dataset/Picture", "train",transforms.ToTensor())
    print(len(x))
    # for i in x:
    #     print(i[0])

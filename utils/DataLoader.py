from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.DataSet import AVEDataset
from utils.test_config import get_parser


def get_dataLoader(args):
    mean_vals = [0.485, 0.456, 0.406]  # 数据均值
    std_vals = [0.229, 0.224, 0.225]  # 数据标准差
    input_size = int(args.input_size)  # 输入大小
    crop_size = int(args.crop_size)  # 裁剪大小
    tsfm_train = transforms.Compose([transforms.Resize((input_size, input_size)),  # 修改分辨率
                                     transforms.RandomCrop(crop_size),  # 随机裁剪
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     # 修改亮度、名度等
                                     transforms.ToTensor(),  # 转变为tensor，并/255
                                     transforms.Normalize(mean_vals, std_vals),  # 归一化
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize((input_size, input_size)),  # 同理
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    if args.is_train:
        img_train = AVEDataset(data_dir=args.Pic_path, mode="train", transform=tsfm_train)
        img_val = AVEDataset(data_dir=args.Pic_path, mode="test", transform=tsfm_train)
        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(img_val, batch_size=8, shuffle=False, num_workers=args.num_workers)
        return train_loader, val_loader
    else:
        img_test = AVEDataset(data_dir=args.Pic_path, mode="test", transform=tsfm_test)
        test_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return test_loader


if __name__ == "__main__":
    test_dl = get_dataLoader(args=get_parser())
    print(get_parser().input_size)
    for id, a in enumerate(test_dl):
        img_name1, img1, inda1, label1 = a
        print(img_name1[0])
        print(img_name1[0][-18:-7])
        print(img1.size())
        break

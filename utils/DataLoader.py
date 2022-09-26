from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.DataSet import AVEDataset
from utils.args_config import get_parser


def get_dataLoader(Pic_path, H5_path, GT_path, train_mode, STA_mode, batch_size, input_size, crop_size):
    mean_vals = [0.485, 0.456, 0.406]  # 数据均值
    std_vals = [0.229, 0.224, 0.225]  # 数据标准差

    tsfm_train = transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.RandomCrop((crop_size, crop_size)),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    tsfm_gt = transforms.Compose([transforms.Resize(crop_size), transforms.ToTensor()])

    if train_mode == "train":
        img_train = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, gt_dir=GT_path, STA_mode=STA_mode,
                               mode="train", transform=tsfm_train, transforms_gt=tsfm_gt)
        train_loader = DataLoader(img_train, batch_size=batch_size, shuffle=True, drop_last=False)
        print("TrainSet.len:", len(img_train), "\t dataLoader.len:", len(train_loader), 'batch_size:', batch_size)
        return train_loader
    elif train_mode == "test":
        img_test = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, gt_dir=GT_path,STA_mode=STA_mode,
                              mode="test", transform=tsfm_test, transforms_gt=tsfm_gt)
        test_loader = DataLoader(img_test, batch_size=batch_size, shuffle=True, drop_last=False)
        print("TestSet.len:", len(img_test), "\t dataLoader.len:", len(test_loader), 'batch_size:', batch_size)
        return test_loader
    elif train_mode == "val":
        img_val = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, gt_dir=GT_path, STA_mode=STA_mode,
                             mode="val", transform=tsfm_test, transforms_gt=tsfm_gt)
        val_loader = DataLoader(img_val, batch_size=batch_size, shuffle=False, drop_last=True)
        print("ValSet.len:", len(val_loader), "\t dataLoader.len:", len(val_loader), 'batch_size:', batch_size)
        return val_loader
    elif train_mode == "att":
        img_all = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, gt_dir=GT_path, STA_mode=STA_mode,
                             mode="all", transform=tsfm_test, transforms_gt=tsfm_gt)
        all_loader = DataLoader(img_all, batch_size=batch_size, shuffle=False, drop_last=True)
        print("ValSet.len:", len(all_loader), "\t dataLoader.len:", len(all_loader), 'batch_size:', batch_size)
        return all_loader


if __name__ == "__main__":
    args = get_parser()
    test_dl = get_dataLoader(Pic_path=args.Pic_path, H5_path=args.H5_path, GT_path=args.GT_path,
                             train_mode="test", STA_mode="SA", batch_size=args.batch_size,
                             input_size=args.input_size, crop_size=args.crop_size)
    i = 0
    for _, a in enumerate(test_dl):

        img_name1, img1, aud1, inda1, label1 = a
        print(img_name1[0])
        print(img_name1[0][-18:-7])
        print(img1.size(), aud1.size())
        i += 1
        if i>3:
            break

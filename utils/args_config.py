import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description='CPL')  # 获得命令行参数

    parser.add_argument("--dataset_name", type=str, default="AVE")
    # ------------------------------ubuntu------------------------------------------
    parser.add_argument("--Data_path", type=str, default=r"/home/ubuntu/AVE_Dataset")
    parser.add_argument("--Video_path", type=str, default=r"/home/ubuntu/AVE_Dataset/Video")
    parser.add_argument("--Pic_path", type=str, default=r"/home/ubuntu/AVE_Dataset/Picture")
    # parser.add_argument("--Pic_path", type=str, default=r"/home/ubuntu/AVE_Dataset/Crop_Picture")    # after_crop
    parser.add_argument("--H5_path", type=str, default=r"/home/ubuntu/AVE_Dataset/H5")
    parser.add_argument("--Att_re_path", type=str, default=r"/media/ubuntu/Data/Result/Att")
    parser.add_argument("--Att_inf_path", type=str, default=r"/media/ubuntu/Data/Result/Att_30")
    parser.add_argument("--Crop_path", type=str, default=r"/home/ubuntu/AVE_Dataset/Crop_Picture")
    parser.add_argument("--GT_path", type=str, default=r"/home/ubuntu/AVE_Dataset/GT")
    parser.add_argument("--save_dir", type=str, default=r'/media/ubuntu/Data/Result/')
    # ------------------------------win--------------------------------------------
    # parser.add_argument("--Data_path", type=str, default=r"D:\WorkPlace\Python\my_STA\AVE_Dataset")
    # parser.add_argument("--Video_path", type=str, default=r"D:\WorkPlace\Python\my_STA\AVE_Dataset\Video")
    # parser.add_argument("--Pic_path", type=str, default=r"D:\WorkPlace\Python\my_STA\AVE_Dataset\Picture")
    # parser.add_argument("--H5_path", type=str, default=r"D:\WorkPlace\Python\my_STA\AVE_Dataset\H5")
    # parser.add_argument("--GT_path", type=str, default=r"D:\WorkPlace\Python\my_STA\AVE_Dataset\GT")
    # parser.add_argument("--save_dir", type=str, default=r'D:\WorkPlace\Python\my_STA\Result')
    # -----------------------------------------------------------------------------
    parser.add_argument("--need_val_repic_save", type=bool, default=False, help="Do you want to save all result_pic when val ?")
    parser.add_argument("--train_mode", type=str, default=r"train", help="train/test/val")
    parser.add_argument("--STA_mode", type=str, default=r"S", help="S/ST/SA/STA")
    parser.add_argument("--val_Pepoch", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64) # train 3090:S-64 SA-40 ST-20 test:2060:S-128 SA:50 ST:64
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--decay_points", type=str, default='5,10')  # 衰变点
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=5)

    print(parser.parse_args())

    return parser.parse_args()


if __name__ == '__main__':
    a = get_parser()

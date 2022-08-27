import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description='CPL')  # 获得命令行参数

    parser.add_argument("--dataset",type=str,default="AVE")
    parser.add_argument("--Data_path", type=str, default=r"../AVE_Dataset")
    parser.add_argument("--Video_path", type=str, default=r"../AVE_Dataset/Video")
    parser.add_argument("--Pic_path", type=str, default=r"../AVE_Dataset/Picture")
    parser.add_argument("--Audio_path", type=str, default=r"../AVE_Dataset/Audio")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--checkpoint_dir", type=str, default='./runs/')
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--decay_points", type=str, default='5,10')  # 衰变点
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=5)

    return parser.parse_args()


if __name__ == '__main__':
    pass

import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(description='CPL')  # 获得命令行参数

    parser.add_argument("--dataset_name",type=str,default="AVE")
    parser.add_argument("--Data_path", type=str, default=r"../AVE_Dataset")
    parser.add_argument("--Video_path", type=str, default=r"../AVE_Dataset/Video")
    parser.add_argument("--Pic_path", type=str, default=r"../AVE_Dataset/Picture")
    parser.add_argument("--Audio_path", type=str, default=r"../AVE_Dataset/Audio")
    parser.add_argument("--train_mode", type=str, default=r"train", help="train/test/val")
    parser.add_argument("--STA_mode", type=str, default=r"S", help="S/ST/SA/STA")
    parser.add_argument("--Checkpoint_dir", type=str, default='./runs/')
    parser.add_argument("--val_Pepoch",type=int,default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--input_size", type=int, default=300)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--decay_points", type=str, default='5,10')  # 衰变点
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--SummaryWriter_dir", type=str, default=r"./log")
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=10)

    print(parser.parse_args())

    return parser.parse_args()


if __name__ == '__main__':
    a = get_parser()
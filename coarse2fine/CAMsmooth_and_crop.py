import os
import os.path as osp
import numpy as np
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.DataFromtxt as dft
from utils.args_config import get_parser


def smooth_and_crop(mode, dataset_path, pic_path, att_path, crop_path):
    data_name_list = dft.readDataTxt(dataset_path, mode)
    for num, data_name in enumerate(data_name_list):
        if data_name[-1] - data_name[-2] <= 3:
            continue
        data_time_range_list = []
        for time_id in range(data_name[-2], data_name[-1]):
            txt_path = osp.join(att_path, data_name[0], "%02d_crop.txt" % time_id)
            with open(txt_path, 'r', encoding='utf-8') as f:
                data_time_range_list.append([int(i) for i in f.readline().strip().split('&')[:-1]])
        data_time_range_list = np.array(data_time_range_list)
        for i in range(data_name[-2], data_name[-1]):
            avg3_smooth = np.array([0, 0, 0, 0])
            for j in range(-1, 2):
                if data_name[-2] <= (i + j) < data_name[-1]:
                    avg3_smooth += data_time_range_list[i + j - data_name[-2]]
                else:
                    avg3_smooth += data_time_range_list[i - data_name[-2]]
            txt_path = osp.join(att_path, data_name[0], "%02d_crop_smooth.txt" % i)
            avg3_smooth = [int(s/3) for s in avg3_smooth.tolist()]
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('&'.join(str(avg) for avg in avg3_smooth))

            min_row, max_row, min_col, max_col = avg3_smooth[0], avg3_smooth[1], avg3_smooth[2], avg3_smooth[3]
            # print(min_row, max_row, min_col, max_col)
            RGB_path = osp.join(pic_path, data_name[0], "%02d" % i + ".jpg")
            RGB = cv2.resize(cv2.imread(RGB_path), (356, 356))
            cut_result = RGB[min_row: max_row + 1, min_col: max_col + 1, :]
            print(cut_result.shape)
            # print(min_row, max_row, min_col, max_col)
            crop_save_path = osp.join(crop_path, data_name[0], "%02d" % i + ".jpg")
            if not os.path.exists(osp.join(crop_path, data_name[0])):
                os.makedirs(osp.join(crop_path, data_name[0]))
            cv2.imwrite(crop_save_path, cut_result)
            print("%4d/%4d  write crop pic to" % (num, len(data_name_list)), crop_save_path)
            RGB_ret = cv2.rectangle(RGB, (min_col, min_row), (max_col, max_row), (0, 0, 255), 8)
            RGB_ret_path = osp.join(att_path, data_name[0], "%02d_ret.jpg" % i)
            cv2.imwrite(RGB_ret_path, RGB_ret)


if __name__ == "__main__":
    args = get_parser()
    smooth_and_crop(mode="train", dataset_path=args.Data_path, pic_path=args.Pic_path,
                    att_path=args.Att_re_path, crop_path=args.Crop_path)
    smooth_and_crop(mode="test", dataset_path=args.Data_path, pic_path=args.Pic_path,
                    att_path=args.Att_re_path, crop_path=args.Crop_path)

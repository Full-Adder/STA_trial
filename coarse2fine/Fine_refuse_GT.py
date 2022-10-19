import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args_config import get_parser
from utils.DataFromtxt import readDataTxt


def MatrixNormalization(M):
    maxValue = np.max(M)
    minValue = np.min(M)
    if maxValue - minValue != 0:
        re = (M - minValue) / (maxValue - minValue)
        return re
    else:
        return M


def refuse(mode, Data_p, pic_path, crop_path, att_dir, gt_dir):
    # train data_path 获取处理图片的位置 裁剪信息位置 使用的CAM SCAM保存的位置
    data_list = readDataTxt(Data_p, mode)
    for num, data in enumerate(data_list):
        if data[-1] - data[-2] <= 3:
            continue
        for id in range(data[-2], data[-1]):
            att_Pic_dir = os.path.join(att_dir, data[0], "%02d" % id)
            Att_pic_S = os.path.join(att_Pic_dir + '_S.png')
            Att_pic_SA = os.path.join(att_Pic_dir + '_SA.png')
            Att_pic_ST = os.path.join(att_Pic_dir + '_ST.png')
            Att_pic_txt = os.path.join(att_Pic_dir + '.txt')

            V = cv2.imread(Att_pic_S)
            A = cv2.imread(Att_pic_SA)
            T = cv2.imread(Att_pic_ST)

            V = MatrixNormalization(V[:, :, 2])
            A = MatrixNormalization(A[:, :, 2])
            T = MatrixNormalization(T[:, :, 2])

            dataSet = dict()
            with open(Att_pic_txt, 'r', encoding="utf-8") as f:
                for d in f.readlines():  # STA_mode 图片名称 预测概率最高的标签 概率 真实标签 预测的真实标签概率
                    d = d.strip().split('&')
                    # dataSet[d[0]] = float(d[3])
                    dataSet[d[0]] = float(d[5])

            if len(dataSet) != 3:
                print("ERROR in data, maybe ST")
                break

            probS, probA, probT = dataSet["S"], dataSet["SA"], dataSet["ST"]

            Up = V * probS + A * probA + T * probT + 0.00001
            Down = probS + probT + probA + 0.00001
            F = MatrixNormalization(Up / Down)

            Pic_dir = os.path.join(pic_path, data[0], "%02d" % id)
            RGB = cv2.resize(cv2.imread(Pic_dir + r'.jpg'), (356, 356))
            crop_txt_path = os.path.join(crop_path, data[0], "%02d" % id)
            with open(crop_txt_path + "_crop_smooth.txt", 'r', encoding='utf-8') as f:
                d_txt = f.readline().strip().split('&')
                min_row, max_row, min_col, max_col = int(d_txt[0]), int(d_txt[1]), int(d_txt[2]), int(d_txt[3])
            heatmap = cv2.resize(F, (max_col - min_col + 1, max_row - min_row + 1))
            mask = np.zeros((356, 356))
            mask[min_row:max_row+1, min_col:max_col+1] = heatmap
            mask = (mask * 255).astype(np.uint8)
            heat_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            SCAM_re = heat_mask * 0.3 + RGB * 0.5

            if not os.path.exists(os.path.join(gt_dir, data[0])):
                os.makedirs(os.path.join(gt_dir, data[0]))
            gt_path = os.path.join(gt_dir, data[0], "%02d" % id)

            cv2.imwrite(gt_path + ".jpg", mask)
            cv2.imwrite(att_Pic_dir + "_gt.jpg", mask)
            cv2.imwrite(att_Pic_dir + "_re_SCAM.jpg", SCAM_re)
            # print("write SCAM pic to", att_Pic_dir + "_re_SCAM.jpg")

        print(num, "/", len(data_list), data[0], "is ok!")


if __name__ == "__main__":
    args = get_parser()
    refuse("train", att_dir=args.Att_re_path, Data_p=args.Data_path, pic_path=args.Pic_path,
           crop_path=args.Crop_path, gt_dir=args.GT_path)
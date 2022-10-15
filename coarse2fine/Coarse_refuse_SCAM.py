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


def generate_crop(mode, Data_p, att_dir):
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

            Up = V * probS + A * probA + T * probT + 0.0001
            Down = probS + probT + probA + 0.0001
            F = MatrixNormalization(Up / Down)

            alpha = 2.0
            [row, col] = np.where(F > alpha * np.mean(F))
            while row.size == 0 or col.size == 0 or np.max(col) - np.min(col) < 100 or np.max(row) - np.min(row) < 100:
                alpha = alpha - 0.05
                [row, col] = np.where(F > alpha * np.mean(F))

            max_col, max_row, min_col, min_row = np.max(col), np.max(row), np.min(col), np.min(row)
            with open(att_Pic_dir + "_crop.txt", 'w', encoding='utf-8') as f:
                f.write('&'.join(str(i) for i in [min_row, max_row, min_col, max_col, alpha]))

            RGB_path = os.path.join(args.Pic_path, data[0], "%02d" % id + ".jpg")
            RGB = cv2.resize(cv2.imread(RGB_path), (356, 356))
            F = (F * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(F, cv2.COLORMAP_JET)
            SCAM_re = heatmap * 0.3 + RGB * 0.5

            # cut_result = RGB[min_row: max_row + 1, min_col: max_col + 1, :]
            # cv2.imwrite(Crop_path + ".jpg", cut_result)
            # print("write crop pic to", Crop_path + ".jpg")
            # SCAM_path = os.path.join(args.SCAM_path, data[0], "%02d" % id)
            # cv2.imwrite(att_Pic_dir + "_crop.jpg", cut_result)
            # print("write crop pic to", att_Pic_dir + "_crop.jpg")

            cv2.imwrite(att_Pic_dir + "_re_SCAM.jpg", SCAM_re)
            cv2.imwrite(att_Pic_dir + "_re_SCAM.png", F)
            print("write SCAM pic to", att_Pic_dir + "_re_SCAM.jpg")

        print(num, "/", len(data_list), data[0], "is ok!")


def test():
    # a = np.random.random([100, 999])
    # a= MatrixNormalization(a)
    # i, j = np.where(a > 2 * np.mean(a))
    # print(i,j)

    img = cv2.imread("../Readme.assets/S_model.png")
    print(type(img), img.shape)
    img = MatrixNormalization(img)

    row, col, k = np.where(img > np.mean(img))
    print(row, col)
    max_col, max_row, min_col, min_row = np.max(col), np.max(row), np.min(col), np.min(row)
    print(max_col, max_row, min_col, min_row)


if __name__ == "__main__":
    # test()

    # att = r"/media/ubuntu/Data/Result/Att_valbA2"
    # crop = r"/media/ubuntu/Data/Result/Crop_Picture_valbA2"

    args = get_parser()
    generate_crop(mode="train", Data_p=args.Data_path, att_dir=args.Att_re_path)
    generate_crop(mode="test", Data_p=args.Data_path, att_dir=args.Att_re_path)

# ls -l|grep "^d"| wc -l 查看当前目录下的文件夹目录个数（不包含子目录中的目录)

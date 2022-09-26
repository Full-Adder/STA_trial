import os
import numpy as np
import cv2
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


if __name__ == "__main__":
    args = get_parser()
    Data_p = args.Data_path
    att_dir = args.Att_inf_path

    data_list = readDataTxt(Data_p, "all")
    for data in data_list:
        for id in range(data[-2], data[-1]):
            att_Pic_dir = os.path.join(att_dir, data[0], "%02d" % id)
            Att_pic_S = os.path.join(att_Pic_dir + '_S.png')
            Att_pic_SA = os.path.join(att_Pic_dir + '_SA.png')
            Att_pic_ST = os.path.join(att_Pic_dir + '_ST.png')
            Att_pic_txt = os.path.join(att_Pic_dir + '.txt')

            V = MatrixNormalization(cv2.imread(Att_pic_S))
            A = MatrixNormalization(cv2.imread(Att_pic_SA))
            T = MatrixNormalization(cv2.imread(Att_pic_ST))

            dataSet = []
            with open(Att_pic_txt, 'r', encoding="utf-8") as f:
                for d in f.readlines():
                    d = d.strip().split('&')
                    dl = [d[0], int(d[1]), float(d[2]), int(d[3])]
                    dataSet.append(dl)

            if len(dataSet) != 3:
                print("ERROR in data, maybe ST")
                break

            probS, probA, probT = dataSet[0][2], dataSet[1][2], dataSet[2][2]

            Up = V * probS + A * probA + T * probT + 0.0001
            Down = probS + probT + probA + 0.0001
            F = MatrixNormalization(Up / Down)

            [row, col] = np.where(F > 2 * np.mean(F))
            max_col, max_row, min_col, min_row = np.max(col), np.max(row), np.min(col), np.min(row)

            RGB_path = os.path.join(args.pic_dir, data[0], "%02d" % id + ".jpg")
            RGB = cv2.resize(cv2.imread(RGB_path), (356, 356))
            result = RGB[min_row: max_row, min_col: max_col]

            Crop_path = os.path.join(args.Crop_path, data[0], "%02d" % id)
            cv2.imwrite(RGB, Crop_path + "_crop.jpg")

            with open(Crop_path + ".txt", "w", encoding='utf-8') as f:
                f.writelines('&'.join(str(i) for i in [max_col, max_row, min_col, min_row]))

        print(data[0], "is ok!")


def test():
    img = cv2.imread("../Readme.assets/S_model.png")
    print(type(img), img.shape)
    a = np.random.random([3, 4])
    print(np.where(a > 0.5))
    print(a)
    print(MatrixNormalization(a))

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


def generate_crop(mode, att_dir=None, crop_path=None):
    args = get_parser()
    Data_p = args.Data_path
    if att_dir is None:
        att_dir = args.Att_inf_pat
    if crop_path is None:
        crop_path = args.Crop_path

    data_list = readDataTxt(Data_p, mode)
    for data in data_list:
        for id in range(data[-2] + 1, data[-1] - 1):
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
                for d in f.readlines():
                    d = d.strip().split('&')
                    dataSet[d[0]] = float(d[2])

            if len(dataSet) != 3:
                print("ERROR in data, maybe ST")
                break

            probS, probA, probT = dataSet["S"], dataSet["SA"], dataSet["ST"]

            Up = V * probS + A * probA + T * probT + 0.0001
            Down = probS + probT + probA + 0.0001
            F = MatrixNormalization(Up / Down)

            [row, col] = np.where(F > 2 * np.mean(F))
            if row.size != 0 and col.size != 0:
                max_col, max_row, min_col, min_row = np.max(col), np.max(row), np.min(col), np.min(row)

                RGB_path = os.path.join(args.Pic_path, data[0], "%02d" % id + ".jpg")
                RGB = cv2.resize(cv2.imread(RGB_path), (356, 356))
                result = RGB[min_row: max_row, min_col: max_col, :]

                if not os.path.exists(os.path.join(crop_path, data[0])):
                    os.makedirs(os.path.join(crop_path, data[0]))
                Crop_path = os.path.join(crop_path, data[0], "%02d" % id)
                cv2.imwrite(Crop_path + ".jpg", result)

                f = open(Crop_path + "_crop.txt", 'w', encoding='utf-8')
                f.write('&'.join(str(i) for i in [min_row, max_row, min_col, max_col]))
                f.close()

        print(data[0], "is ok!")


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
    mode = "all"
    att = r"/media/ubuntu/Data/Result/Att_50"
    crop = r"/media/ubuntu/Data/Result/Crop_Picture_50"
    generate_crop(mode, txt_name=tname, att_dir=att, crop_path=crop)
    # generate_crop("train")
    # generate_crop("val")

# ls -l|grep "^d"| wc -l 查看当前目录下的文件夹目录个数（不包含子目录中的目录)

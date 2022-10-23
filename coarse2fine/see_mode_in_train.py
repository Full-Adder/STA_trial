import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import os
import sys

sys.path.append(r"./STA/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args_config import get_parser
from utils.DataLoader import get_dataLoader
from for_other_dataSet.OtherDataSet import get_otherDataLoader
from utils.model_tool import get_model
from utils.DataFromtxt import id_category

args = get_parser()


def test(model, dat_loader, STA_mode, Pic_path, save_index, save_path_hh):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    with torch.no_grad():
        for id, dat_test in enumerate(dat_loader):
            print(id, ":", len(dat_loader))
            if STA_mode == "S":
                img_name, img1, class_id, onehot_label = dat_test
                img1 = img1.to(device)
                x11, x22, map1, map2 = model(img1)

            elif STA_mode == "SA":
                img_name, img1, aud1, class_id, onehot_label = dat_test
                img1 = img1.to(device)
                aud1 = aud1.to(device)
                x11, x22, map1, map2 = model(img1, aud1)

            elif STA_mode == "ST":
                img_name, img_bef, img_now, img_aft, class_id, onehot_label = dat_test
                img_bef = img_bef.to(device)
                img_now = img_now.to(device)
                img_aft = img_aft.to(device)
                x11, x1, x22, x2, x33, x3, map1, map2 = model(img_bef, img_now, img_aft)

            else:
                img_name, img_bef, aud_bef, gt_bef, img_1, aud_now, gt_now, \
                img_aft, aud_aft, gt_aft, class_id, onehot_label = dat_test

                img_bef = img_bef.to(device)
                img_now = img_1.to(device)
                img_aft = img_aft.to(device)
                aud_bef = aud_aft.to(device)
                aud_now = aud_now.to(device)
                aud_aft = aud_aft.to(device)
                audiocls = torch.load(r'STA/AudioSwitch.pt')
                audiocls.cuda().eval()
                with torch.no_grad():
                    switch_bef = audiocls(aud_bef, img_bef)
                    switch_now = audiocls(aud_now, img_now)
                    switch_aft = audiocls(aud_aft, img_aft)

                p04, p03, p02, p14, p13, p12, p24, p23, p22 = \
                    model(img_bef, img_now, img_aft, aud_bef, aud_now, aud_aft,
                          switch_bef, switch_now, switch_aft)

            if STA_mode != "STA":
                h_x = F.softmax(x11, dim=1).data.squeeze()  # softmax 转化为概率
                probs, index_of_pic = h_x.sort(1, True)  # 1行排序
                index_of_pic = index_of_pic[:, 0]  # 排序后最大值索引

                ind = torch.nonzero(onehot_label)  # [10, 28] -> 非0元素的行列索引

            for i in range(len(img_name)):  # 非0元素的个数
                if STA_mode != "STA":
                    batch_index, la = ind[i]  # 帧索引，类别索引

                    save_accu_map_folder = os.path.join(save_path_hh, "%02d_%s" % (la, id_category[la]),
                                                        img_name[i][:-3])
                    if not os.path.exists(save_accu_map_folder):
                        os.makedirs(save_accu_map_folder)
                    save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-2:] + "_%02d" % save_index)
                    if la != index_of_pic[i]:
                        save_accu_map_path += r"(wrong_%02d_%s)" % (index_of_pic[i], id_category[index_of_pic[i]])
                    atts = (map1[i] + map2[i]) / 2
                    atts[atts < 0] = 0
                    att = atts[la].cpu().data.numpy()  # 转为numpy数组
                else:
                    save_accu_map_folder = os.path.join(save_path_hh, img_name[i][:-3])
                    if not os.path.exists(save_accu_map_folder):
                        os.makedirs(save_accu_map_folder)
                    save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-2:] + "_%02d" % save_index)
                    atts = F.sigmoid(p12[i, 0])
                    print(atts)
                    att = atts.cpu().data.numpy()

                att = cv2.resize(att, (356, 356))
                att = np.rint(att / (att.max() + 1e-10) * 255)  # 归一化到0-255
                att = np.array(att, np.uint8)
                print(att)

                heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                img = cv2.imread(os.path.join(Pic_path, img_name[i] + ".jpg"))
                img = cv2.resize(img, (356, 356))
                result = heatmap * 0.3 + img * 0.5
                cv2.imwrite(save_accu_map_path + ".jpg", result)


def load_model_weight_bef_test(test_weight_id=None, STA_mode="S"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = get_model(STA_mode)
    save_weight_fold = os.path.join(args.save_dir, STA_mode, './model_weight/')
    if args.GT_path != "None":
        test_loader = get_dataLoader(Pic_path=args.Pic_path, H5_path=args.H5_path, GT_path=args.GT_path,
                                     train_mode="train", STA_mode=STA_mode,
                                     batch_size=args.batch_size, input_size=args.input_size)  # 获取测试集
    else:
        test_loader = get_otherDataLoader(Pic_path=args.Pic_path, H5_path=args.H5_path,
                                          batch_size=args.batch_size, input_size=args.input_size)

    test_result_dir = os.path.join(args.save_dir, STA_mode, "pic_re")
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    best_pth = os.path.join(save_weight_fold, "%s_%s_%03d.pth" % (args.dataset_name, STA_mode, test_weight_id))
    if os.path.exists(best_pth):
        print("-----> find pretrained model weight in", best_pth)
        state = torch.load(best_pth)
        net.load_state_dict(state)
    else:
        print("Error! There is not pretrained weight --", test_weight_id, " in", best_pth)

    print("---------> let's test!  -------------->")
    net.to(device)

    test(dat_loader=test_loader, model=net, STA_mode=STA_mode, Pic_path=args.Pic_path,
         save_index=test_weight_id, save_path_hh=test_result_dir)
    # ========================================================================


if __name__ == '__main__':
    # args = get_parser()
    epoch = args.epoch
    load_model_weight_bef_test(test_weight_id=epoch, STA_mode=args.STA_mode)

    # for i in range(31, 32):
    #     print("now let's test weight", i)
    #     load_model_weight_bef_test(i)

#  tensorboard.exe --logdir ./ --samples_per_plugin images=100

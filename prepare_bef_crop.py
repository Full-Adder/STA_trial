import torch
import numpy as np
import torch.nn.functional as F
import cv2
import os
from utils.DataLoader import get_dataLoader
from utils.model_tool import get_model
from utils.args_config import get_parser

args = get_parser()


def generate_att(STA_mode=args.STA_mode, Pic_path=args.Pic_path, H5_path=args.H5_path,
                 batch_size=args.batch_size, input_size=args.input_size,
                 att_dir=args.Att_re_path, model_train_epoch=12):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_weight_fold = os.path.join(args.save_dir, STA_mode, './model_weight/')

    model, _ = get_model(STA_mode)
    best_pth = os.path.join(save_weight_fold, "%s_%s_%03d" %
                            (args.dataset_name, STA_mode, model_train_epoch) + '.pth')
    if os.path.exists(best_pth):
        print("-----> find pretrained model weight in", best_pth)
        state = torch.load(best_pth)
        model.load_state_dict(state)
    else:
        print("can't find weight in ", best_pth)
        exit()
    model.to(device)
    model.eval()

    train_loader = get_dataLoader(Pic_path=Pic_path, H5_path=H5_path, GT_path=None,
                                  train_mode="att", STA_mode=STA_mode,
                                  batch_size=batch_size, input_size=input_size, crop_size=256)

    for idx_test, dat_test in enumerate(train_loader):
        with torch.no_grad():
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
                img1 = img_now.to(device)
                img_aft = img_aft.to(device)
                x11, x1, x22, x2, x33, x3, map1, map2 = model(img_bef, img1, img_aft)

            h_x = F.softmax(x11, dim=1).data.squeeze()  # softmax 转化为概率
            probs, index_of_pic = h_x.sort(1, True)  # 1行排序
            probs = probs[:, 0]  # 排序后最大数值
            index_of_pic = index_of_pic[:, 0]  # 排序后最大值索引

            ind = torch.nonzero(onehot_label)  # [10, 28] -> 非0元素的行列索引

            for i in range(img1.shape[0]):  # 非0元素的个数
                batch_index, la = ind[i]  # 帧索引，类别索引

                save_accu_map_folder = os.path.join(att_dir, img_name[i][:-2])
                if not os.path.exists(save_accu_map_folder):
                    os.makedirs(save_accu_map_folder)
                save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-2:])
                atts = (map1[i] + map2[i]) / 2
                atts[atts < 0] = 0

                att = atts[la].cpu().data.numpy()
                att = np.rint(att / (att.max() + 1e-8) * 255)
                att = np.array(att, np.uint8)
                att = cv2.resize(att, (356, 356))
                cv2.imwrite(save_accu_map_path + '_' + STA_mode + '.png', att)

                heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                img = cv2.imread(os.path.join(Pic_path, img_name[i] + ".jpg"))
                img = cv2.resize(img, (356, 356))
                result = heatmap * 0.3 + img * 0.5
                cv2.imwrite(save_accu_map_path + '_re_' + STA_mode + ".jpg", result)

                txt_file = open(save_accu_map_path + '.txt', 'a')
                txt_file.write('&'.join([STA_mode, img_name[i], str(index_of_pic[i].cpu().numpy()),
                                        str(probs[i].cpu().numpy()), str(class_id[i].numpy())])+'\n')
                # 图片名称 预测标签 预测概率 标签
                txt_file.close()
                print("has save", save_accu_map_path)


if __name__ == "__main__":
    # =========================== val_best =============================
    # generate_att(STA_mode="S", batch_size=550, model_train_epoch=11)
    # generate_att(STA_mode="ST", batch_size=170, model_train_epoch=9)
    # generate_att(STA_mode="SA", batch_size=500, model_train_epoch=12)

    # =========================== maybe best (+1) =============================
    generate_att(STA_mode="S", batch_size=550, model_train_epoch=12, att_dir="/media/ubuntu/Data/Result/Att_valbA1")
    generate_att(STA_mode="ST", batch_size=170, model_train_epoch=10, att_dir="/media/ubuntu/Data/Result/Att_valbA1")
    generate_att(STA_mode="SA", batch_size=500, model_train_epoch=13, att_dir="/media/ubuntu/Data/Result/Att_valbA1")

    # ========================== maybe best (+2) ============================
    generate_att(STA_mode="S", batch_size=550, model_train_epoch=13, att_dir="/media/ubuntu/Data/Result/Att_valbA2")
    generate_att(STA_mode="ST", batch_size=170, model_train_epoch=11, att_dir="/media/ubuntu/Data/Result/Att_valbA2")
    generate_att(STA_mode="SA", batch_size=500, model_train_epoch=14, att_dir="/media/ubuntu/Data/Result/Att_valbA2")

    # ========================== propose best (30) ============================
    generate_att(STA_mode="S", batch_size=550, model_train_epoch=30, att_dir="/media/ubuntu/Data/Result/Att_30")
    generate_att(STA_mode="ST", batch_size=170, model_train_epoch=30, att_dir="/media/ubuntu/Data/Result/Att_30")
    generate_att(STA_mode="SA", batch_size=500, model_train_epoch=30, att_dir="/media/ubuntu/Data/Result/Att_30")

    # ========================== true best (30) ============================
    generate_att(STA_mode="S", batch_size=550, model_train_epoch=50, att_dir="/media/ubuntu/Data/Result/Att_50")
    generate_att(STA_mode="ST", batch_size=170, model_train_epoch=50, att_dir="/media/ubuntu/Data/Result/Att_50")
    generate_att(STA_mode="SA", batch_size=500, model_train_epoch=50, att_dir="/media/ubuntu/Data/Result/Att_50")

    pass

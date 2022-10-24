import torch
import numpy as np
import torch.nn.functional as F
import cv2
import os
import sys

sys.path.append(r"./STA/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args_config import get_parser
from for_other_dataSet.OtherDataSet import get_otherDataLoader
from utils.model_tool import get_model

args = get_parser()


def test(model, dat_loader, Pic_path, save_path_hh):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    with torch.no_grad():
        for id, dat_test in enumerate(dat_loader):
            print(id, ":", len(dat_loader))
            img_name, img_bef, aud_bef, img_1, aud_now, img_aft, aud_aft = dat_test

            img_bef = img_bef.to(device)
            img_now = img_1.to(device)
            img_aft = img_aft.to(device)
            aud_bef = aud_aft.to(device)
            aud_now = aud_now.to(device)
            aud_aft = aud_aft.to(device)

            # audiocls = torch.load(r'STA/AudioSwitch.pt')
            # audiocls.cuda().eval()
            # with torch.no_grad():
            #     switch_bef = audiocls(aud_bef, img_bef)
            #     switch_now = audiocls(aud_now, img_now)
            #     switch_aft = audiocls(aud_aft, img_aft)
            #
            # p04, p03, p02, p14, p13, p12, p24, p23, p22 = model(img_bef, img_now, img_aft, aud_bef, aud_now, aud_aft,
            #                                                     switch_bef, switch_now, switch_aft)

            map0, map0_1, map1, map1_1, map2, map2_1 = model(img_bef, img_now, img_aft, aud_bef, aud_now, aud_aft)



            for i in range(len(img_name)):  # 非0元素的个数
                save_accu_map_folder = os.path.join(save_path_hh, img_name[i][:-5])
                if not os.path.exists(save_accu_map_folder):
                    os.makedirs(save_accu_map_folder)
                save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-4:])
                atts = F.sigmoid(map1[i, 0])
                # print(atts)
                att = atts.cpu().data.numpy()

                att = cv2.resize(att, (356, 356))
                att = np.rint(att / (att.max() + 1e-10) * 255)  # 归一化到0-255
                att = np.array(att, np.uint8)
                # print(att)

                heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                img = cv2.imread(os.path.join(Pic_path, img_name[i] + ".jpg"))
                img = cv2.resize(img, (356, 356))
                result = heatmap * 0.3 + img * 0.5
                cv2.imwrite(save_accu_map_path + ".jpg", result)
                print("write to", save_accu_map_path)


def load_model_weight_bef_test(test_weight_id=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = get_model("STA")
    save_weight_fold = os.path.join(args.save_dir, "STA", './model_weight/')
    test_loader = get_otherDataLoader(Pic_path=args.Pic_path, H5_path=args.H5_path,
                                      batch_size=args.batch_size, input_size=args.input_size)

    best_pth = os.path.join(save_weight_fold, "STA_%03d.pth" % test_weight_id)
    if os.path.exists(best_pth):
        print("-----> find pretrained model weight in", best_pth)
        state = torch.load(best_pth)
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(state)
    else:
        print("Error! There is not pretrained weight --", test_weight_id, " in", best_pth)

    print("---------> let's test!  -------------->")
    net.to(device)

    test(dat_loader=test_loader, model=net, Pic_path=args.Pic_path, save_path_hh=args.Att_re_path)
    # ========================================================================


if __name__ == '__main__':
    # args = get_parser()
    epoch = args.epoch
    load_model_weight_bef_test(test_weight_id=epoch)


#  tensorboard.exe --logdir ./ --samples_per_plugin images=100

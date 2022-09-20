import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import os
from loss import KLDLoss
from utils.args_config import get_parser
from torch.utils.tensorboard import SummaryWriter
from utils.DataLoader import get_dataLoader
from datetime import datetime
from utils.model_tool import get_model


def test(model, Pic_path, H5_path, is_val, save_index,
         batch_size, input_size, dataset_name, Summary_Writer, test_re_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.eval()
    loss2 = nn.BCEWithLogitsLoss()
    loss1 = KLDLoss()
    audiocls = torch.load('27001.pt')
    audiocls.cuda().eval()

    if is_val:
        save_path_hh = os.path.join(test_re_dir, str(save_index))
    else:
        save_path_hh = test_re_dir

    val_mode = "val" if is_val else "test"

    test_loader = get_dataLoader(Pic_path=Pic_path, H5_path=H5_path, train_mode=val_mode, STA_mode="STA",
                                 batch_size=batch_size, input_size=input_size)  # 获取测试集

    for idx_test, dat_test in enumerate(test_loader):

        with torch.no_grad():
            img_name, \
            img_bef, aud_bef, gt_bef, \
            img_now, aud_now, gt_now, \
            img_aft, aud_aft, gt_aft, \
            class_id, onehot_label = dat_test

            img_bef = img_bef.to(device)
            img_now = img_now.to(device)
            img_aft = img_aft.to(device)
            aud_bef = aud_aft.to(device)
            aud_now = aud_now.to(device)
            aud_aft = aud_aft.to(device)
            gt_bef = gt_bef.to(device)
            gt_now = gt_now.to(device)
            gt_aft = gt_aft.to(device)
            with torch.no_grad():
                switch_bef = audiocls(aud_bef, img_bef)
                switch_now = audiocls(aud_now, img_now)
                switch_aft = audiocls(aud_aft, img_aft)
                # class_id = class_id.to(device)

            p04, p03, p02, p14, p13, p12, p24, p23, p22 = \
                model(img_bef, img_now, img_aft, aud_bef, aud_now, aud_aft,
                      switch_bef, switch_now, switch_aft)

            loss_t = loss2(p04, gt_bef) + loss2(p14, gt_now) + loss2(p24, gt_aft) + \
                     loss2(p03, gt_bef) + loss2(p13, gt_now) + loss2(p23, gt_aft) + \
                     loss2(p02, gt_bef) + loss2(p12, gt_now) + loss2(p22, gt_aft) + \
                     loss1(F.sigmoid(p04), gt_bef) + loss1(F.sigmoid(p14), gt_now) + \
                     loss1(F.sigmoid(p24), gt_aft) + loss1(F.sigmoid(p03), gt_bef) + \
                     loss1(F.sigmoid(p13), gt_now) + loss1(F.sigmoid(p23), gt_aft) + \
                     loss1(F.sigmoid(p02), gt_bef) + loss1(F.sigmoid(p12), gt_now) + \
                     loss1(F.sigmoid(p22), gt_aft)

        result_show_list = []

        if is_val:
            Summary_Writer.add_scalars(dataset_name, {args.STA_mode + "_val_loss": loss_t.data.item()},
                                       (save_index * len(test_loader) + idx_test) * 8)
        else:
            Summary_Writer.add_scalar(dataset_name + "_" + args.STA_mode + "_test_loss", loss_t.data.item(),
                                      save_index * len(test_loader) + idx_test)

        dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        if idx_test % 10 == 0:
            print('time:{}\t'
                  'Batch: [{:4d}/{:4d}]\t'
                  'Loss {:.4f})\t'.format(dt, idx_test, len(test_loader), loss_t.data.item()))

        for i in range(p12.size()[0]):  # batch_size
            save_accu_map_folder = os.path.join(save_path_hh, img_name[i][:-3])
            if not os.path.exists(save_accu_map_folder):
                os.makedirs(save_accu_map_folder)
            save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-2:])
            att = F.sigmoid(p12[i][0]).cpu().data.numpy()
            att = np.rint(att / (att.max() + 1e-8) * 255)  # 归一化到0-255
            att = np.array(att, np.uint8)
            att = cv2.resize(att, (220, 220))  # 修改分辨率

            cv2.imwrite(save_accu_map_path + '.png', att)  # 保存图片

            heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
            img = cv2.imread(os.path.join(Pic_path, img_name[i] + ".jpg"))
            img = cv2.resize(img, (356, 356))
            result = heatmap * 0.3 + img * 0.5
            cv2.imwrite(save_accu_map_path + ".jpg", result)

            if is_val and idx_test % (len(test_loader) // 4) == 0:
                img = cv2.imread(save_accu_map_path + ".jpg")
                img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result_show_list.append(img1)

        if len(result_show_list) > 0:
            Summary_Writer.add_images("result batch:" + str(idx_test), np.stack(result_show_list, 0),
                                      save_index, dataformats="NHWC")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_parser()
    net, _ = get_model("STA")
    save_weight_fold = os.path.join(args.save_dir, args.STA_mode, './model_weight/')

    test_epoch = input(r"please input your test weight of your mode! Maybe is 10/20/.../best")
    if test_epoch == "best":
        best_pth = os.path.join(save_weight_fold, '%s_%s_model_bast.pth.tar' % (args.dataset_name, args.STA_mode))
        if os.path.exists(best_pth):
            print("-----> find pretrained model weight in", best_pth)
            state = torch.load(best_pth)
            net.load_state_dict(state['state_dict'])
        else:
            print("Error! There is not pretrained weight --" + test_epoch, "in", best_pth)
            exit()

    else:
        num = int(test_epoch)
        best_pth = os.path.join(save_weight_fold, "%s_%s_%03d" % (args.dataset_name, args.STA_mode, num) + '.pth')
        if os.path.exists(best_pth):
            print("-----> find pretrained model weight in", best_pth)
            state = torch.load(best_pth)
            net.load_state_dict(state)
        else:
            print("Error! There is not pretrained weight --" + test_epoch, "in", best_pth)
            exit()

    print("-----> success load pretrained weight form ", best_pth)
    print("-----> let's test! -------------->")
    net.to(device)

    # ============================val()===================================
    # writer = SummaryWriter(r'./test_result/log')
    # test(model=net, Pic_path=args.Pic_path, is_val=False, save_index=0, batch_size=args.batch_size,
    #      input_size=args.input_size, dataset_name=args.dataset_name, Summary_Writer=writer,
    #      test_re_dir=r'./test_result')

    # =============================test()====================================
    test_result_dir = os.path.join("/media/ubuntu/Data", r'./STA_test_result/')
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    writer = SummaryWriter(os.path.join(test_result_dir, r'./log/'))
    test(model=net, Pic_path=args.Pic_path, H5_path=args.H5_path, is_val=False,
         save_index=0, batch_size=args.batch_size, input_size=args.input_size,
         dataset_name=args.dataset_name, Summary_Writer=writer, test_re_dir=test_result_dir)
    writer.close()

#  tensorboard --logdir ./ --samples_per_plugin images=100

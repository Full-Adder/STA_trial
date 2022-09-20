import torch
import numpy as np
import torch.nn.functional as F
import cv2
import os
from utils.args_config import get_parser
from torch.utils.tensorboard import SummaryWriter
from utils.DataLoader import get_dataLoader
from datetime import datetime
from utils.model_tool import get_model
from utils.DataFromtxt import id_category


def test(model, Pic_path, is_val, save_index, batch_size,
         input_size, dataset_name, Summary_Writer, test_re_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    if is_val:
        save_path_hh = os.path.join(test_re_dir, str(save_index))
    else:
        save_path_hh = test_re_dir
    val_mode = "val" if is_val else "test"
    test_loader = get_dataLoader(Pic_path=Pic_path, train_mode=val_mode, STA_mode="S",
                                 batch_size=batch_size, input_size=input_size)  # 获取测试集
    # !!
    for idx_test, dat_test in enumerate(test_loader):
        with torch.no_grad():
            img_name, img1, class_id, onehot_label = dat_test
            class_id = class_id.to(device)
            img1 = img1.to(device)

            x11, x22, map1, map2 = model(img1)
            loss_t = F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)

        result_show_list = []

        if is_val:
            Summary_Writer.add_scalars(dataset_name, {"val_loss": loss_t.data.item()},
                                       (save_index * len(test_loader) + idx_test) * 8)
        else:
            Summary_Writer.add_scalar(dataset_name + "_" + args.STA_mode + "_test_loss", loss_t.data.item(),
                                      save_index * len(test_loader) + idx_test)

        dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        if idx_test % args.disp_interval == 0:
            print('time:{}\t'
                  'Batch: [{:4d}/{:4d}]\t'
                  'Loss {:.4f})\t'.format(dt, idx_test, len(test_loader), loss_t.data.item()))

        if not is_val or args.need_val_repic_save or (idx_test % (len(test_loader) // 4) == 0):
            # is_test or you need to save all val_repic
            # or (idx_test % (len(test_loader)//4)==0 which need to generate tensorboard pic)
            h_x = F.softmax(x11, dim=1).data.squeeze()  # softmax 转化为概率
            probs, index_of_pic = h_x.sort(1, True)  # 1行排序         # 降序
            probs = probs[:, 0]  # 排序后最大数值
            index_of_pic = index_of_pic[:, 0]  # 排序后最大值索引

            ind = torch.nonzero(onehot_label)  # [10, 28] -> 非0元素的行列索引

            for i in range(ind.shape[0]):  # 非0元素的个数
                batch_index, la = ind[i]  # 帧索引，类别索引

                save_accu_map_folder = os.path.join(save_path_hh, "%02d_%s" % (la, id_category[la]), img_name[i][:-3])
                if not os.path.exists(save_accu_map_folder):
                    os.makedirs(save_accu_map_folder)
                save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-2:])
                if la != index_of_pic[i]:
                    save_accu_map_path += r"(wrong_%02d_%s)" % (index_of_pic[i], id_category[index_of_pic[i]])
                atts = (map1[i] + map2[i]) / 2  # 计算两幅图的平均值
                atts[atts < 0] = 0

                att = atts[la].cpu().data.numpy()  # 转为numpy数组
                att = np.rint(att / (att.max() + 1e-8) * 255)  # 归一化到0-255
                att = np.array(att, np.uint8)
                att = cv2.resize(att, (220, 220))  # 修改分辨率
                cv2.imwrite(save_accu_map_path + '.png', att)  # 保存图片

                heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)
                img = cv2.imread(os.path.join(Pic_path, img_name[i] + ".jpg"))
                img = cv2.resize(img, (220, 220))
                result = heatmap * 0.3 + img * 0.5
                if (not is_val) or args.need_val_repic_save:
                    cv2.imwrite(save_accu_map_path + '.png', att)  # 保存图片
                    cv2.imwrite(save_accu_map_path + ".jpg", result)
                if is_val:
                # if True:
                    cv2.imwrite(save_accu_map_path + ".jpg", result)
                    img = cv2.imread(save_accu_map_path + ".jpg")
                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result_show_list.append(img1)

        if len(result_show_list) > 0:
            Summary_Writer.add_images("result batch:" + str(idx_test), np.stack(result_show_list, 0),
                                      save_index, dataformats="NHWC")


def load_model_weight_bef_test(test_weight_id=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = get_model(args.STA_mode)
    save_weight_fold = os.path.join(args.save_dir, args.STA_mode, './model_weight/')

    if test_weight_id == -1:
        test_epoch = input(r"please input your test weight of your mode! Maybe is 10/20/.../best")
    else:
        test_epoch = str(test_weight_id)

    if test_epoch == "best":
        best_pth = os.path.join(save_weight_fold, '%s_%s_model_best.pth.tar' % (args.dataset_name, args.STA_mode))
        if os.path.exists(best_pth):
            print("-----> find pretrained model weight in", best_pth)
            state = torch.load(best_pth)
            net.load_state_dict(state['state_dict'])
            test_epoch = str(state['epoch'] + 1) + "_best"
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

    # ============================val()======================================
    # writer = SummaryWriter(r'./test_result/log')
    # test(model=net, Pic_path=args.Pic_path, is_val=False, save_index=0, batch_size=args.batch_size,
    #      input_size=args.input_size, dataset_name=args.dataset_name, Summary_Writer=writer,
    #      test_re_dir=r'./test_result')

    # =============================test()====================================
    test_result_dir = os.path.join(args.save_dir, args.STA_mode, r'./%s_test_result_%s/' % (args.STA_mode, test_epoch))  # 结果保存文件夹
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, args.STA_mode, r'./test_log/', r'./%s_test_result_%s/' % (args.STA_mode, test_epoch)))
    test(model=net, Pic_path=args.Pic_path, is_val=False,
         save_index=0, batch_size=args.batch_size, input_size=args.input_size,
         dataset_name=args.dataset_name, Summary_Writer=writer, test_re_dir=test_result_dir + r"/pic_result/")
    writer.close()


if __name__ == '__main__':
    args = get_parser()
    load_model_weight_bef_test(30)

    # for i in range(1, 29):
    #     print("now let's test weight", i)
    #     load_model_weight_bef_test(i)

#  tensorboard.exe --logdir ./ --samples_per_plugin images=100

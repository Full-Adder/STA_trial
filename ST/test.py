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


def test(model, Pic_path, is_val, save_index,
         batch_size, input_size, dataset_name, Summary_Writer, test_re_dir=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.eval()

    if is_val:
        save_path_hh = './val_re/' + str(save_index)
    else:
        save_path_hh = test_re_dir

    test_loader = get_dataLoader(Pic_path=Pic_path, train_mode="test", STA_mode="ST",
                                 batch_size=batch_size, input_size=input_size)  # 获取测试集

    for idx_test, dat_test in enumerate(test_loader):

        with torch.no_grad():
            img_name, img_bef, img_now, img_aft, class_id, onehot_label = dat_test

            img_bef = img_bef.to(device)
            img_now = img_now.to(device)
            img_aft = img_aft.to(device)
            class_id = class_id.to(device)

            x11, x1, x22, x2, x33, x3, map1, map2 = model(img_bef, img_now, img_aft)

            loss_t = 0.4 * (F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)
                                + F.cross_entropy(x33, class_id)) \
                         + 0.6 * (F.cross_entropy(x1, class_id) + F.cross_entropy(x2, class_id)
                                  + F.cross_entropy(x3, class_id))

        result_show_list = []
        if is_val:
            Summary_Writer.add_scalars(dataset_name, {"val_loss": loss_t.data.item()},
                                       (save_index * len(test_loader) + idx_test) * 8)

        dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        if idx_test % 10 == 0:
            print('time:{}\t'
                  'Batch: [{:4d}/{:4d}]\t'
                  'Loss {:.4f})\t'.format(dt, idx_test, len(test_loader), loss_t.data.item()))

        h_x = F.softmax(x11, dim=1).data.squeeze()  # softmax 转化为概率
        probs, index_of_pic = h_x.sort(1, True)  # 1行排序                     #降序
        probs = probs[:, 0]  # 排序后最大数值
        index_of_pic = index_of_pic[:, 0]  # 排序后最大值索引

        ind = torch.nonzero(onehot_label)  # [10, 28] -> 非0元素的行列索引

        for i in range(ind.shape[0]):  # batch
            batch_index, la = ind[i]  # 帧索引，类别索引

            if is_val or la == index_of_pic[i]:  # 如果标签中非0索引==最大结果的索引
                save_accu_map_folder = os.path.join(save_path_hh, img_name[i][:-3])
                if not os.path.exists(save_accu_map_folder):
                    os.makedirs(save_accu_map_folder)
                save_accu_map_path = os.path.join(save_accu_map_folder, img_name[i][-2:])
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
                cv2.imwrite(save_accu_map_path + ".jpg", result)

                if is_val and idx_test % (len(test_loader) // 4) == 0:
                    img = cv2.imread(save_accu_map_path + ".jpg")
                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result_show_list.append(img1)

        if len(result_show_list) > 0:
            Summary_Writer.add_images("result batch:" + str(idx_test), np.stack(result_show_list, 0),
                                      save_index, dataformats="NHWC")


if __name__ == '__main__':
    args = get_parser()
    net, _ = get_model("S")
    best_pth = "./runs/model_best.pth.tar"
    if os.path.exists(best_pth):
        state = torch.load(best_pth)
        net.load_state_dict(state['state_dict'])
        print("load pretrained weight form ", best_pth)

    if not os.path.exists(r'./test_result'):
        os.makedirs(r'./test_result')
    net.to("cuda")

    # test()
    # writer = SummaryWriter(r'./test_result/log')
    # test(model=net, Pic_path=args.Pic_path, is_val=False, save_index=0, batch_size=args.batch_size,
    #      input_size=args.input_size, dataset_name=args.dataset_name, Summary_Writer=writer,
    #      test_re_dir=r'./test_result')

    # val()
    writer = SummaryWriter(r'./log')
    for i in range(5):
        test(model=net, Pic_path=args.Pic_path, is_val=True, save_index=i, batch_size=args.batch_size,
             input_size=args.input_size, dataset_name=args.dataset_name, Summary_Writer=writer)
    writer.close()

#  tensorboard.exe --logdir ./ --samples_per_plugin images=100

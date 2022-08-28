import os
import torch
import numpy as np
import time
import cv2
import shutil       # 文件操作
import my_optim     # 作者的优化器
import torch.optim as optim
from Smodel import SNetModel
import torch.nn.functional as F
from utils.test_config import get_parser
from utils.DataLoader import get_dataLoader
from utils.avgMeter import AverageMeter
from utils.AVE_modify import modefy_data
from datetime import datetime


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):   # 函数：保存检查点（命令行参数，状态，是否是最优，文件名）
    savepath = os.path.join(args.snapshot_dir, filename)                    # 从命令行参数中获取快照存储路径与文件名组成存储路径
    torch.save(state, savepath)                                             # torch.save 方法及将 state 保存在 savepath
    if is_best:                                                             # 如果是最好的
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))    # 文件命名best复制到快照文件夹


def get_model(args):                                # 获取 model
    model = SNetModel()                             # 获得S
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                    # GPU
    model = torch.nn.DataParallel(model).cuda()
    model.to(device)                                # 加载到GPU:0
    param_groups = model.module.get_parameter_groups()  # 获取 1作者定义的网络的权重 2作者定义的网罗的偏置 3resnext的权重 4resnext的偏置
    optimizer = optim.SGD([                                 # 随机梯度下降
        {'params': param_groups[0], 'lr': args.lr},         # 对于不同类型参数用不同的学习率
        {'params': param_groups[1], 'lr': 2 * args.lr},
        {'params': param_groups[2], 'lr': 10 * args.lr},
        {'params': param_groups[3], 'lr': 20 * args.lr}], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    return model, optimizer                             # 返回模型和优化器


def train(args, save_index):
    batch_time = AverageMeter()
    losses = AverageMeter()

    total_epoch = args.epoch                # 总轮数
    global_counter = args.global_counter    # 全局计数器
                          # 当前轮数

    train_loader, val_loader = get_dataLoader(args)  # data_load--> 训练集 验证集
    model, optimizer = get_model(args)                  # model 和 优化器

    end = time.time()

    epoch = 0
    for epoch in range(total_epoch):                  # 当前轮数小于总共需要的轮数
        model.train()                                   # train
        losses.reset()                                  # reset()置零
        batch_time.reset()

        for idx, dat in enumerate(train_loader):    # 从train_loader中获取数据

            img_name1, img1, inda1, label1 = dat
            label1 = label1.cuda(non_blocking=True)

            x11, x22, map1, map2 = model(img1)
            loss_train = F.multilabel_soft_margin_loss(x11, label1) + F.multilabel_soft_margin_loss(x22, label1)
            # loss

            optimizer.zero_grad()   # 梯度置0
            loss_train.backward()   # 反向传播
            optimizer.step()        # 优化一次

            losses.update(loss_train.data.item(), img1.size()[0])   # 计算平均的loss
            batch_time.update(time.time() - end)                    # 计算每个batch的平均时间
            end = time.time()                                       # 更新结束时间

            global_counter += 1

            if global_counter % args.disp_interval == 0:           # 该打印输出了
                dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
                print('time:{}\t'
                      'Epoch: [{:2d}][{:4d}/{:4d}]\t'
                      'LR: {:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    dt,epoch, global_counter % len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], loss=losses))
            if global_counter % 300 == 0:                           # 500个轮次验证一次
                print("--------------------------------val-------------------------")
                with torch.no_grad():                               # 不计梯度加快验证
                    save_index = save_index + 1
                    for idx_test, dat_test in enumerate(val_loader):    # 取验证集数据
                        model.eval()                                    # 评估模式
                        img_name1, img1, inda1, label1 = dat_test
                        label1 = label1.cuda(non_blocking=True)

                        x11, x22, map1, map2 = model(img1)  # 前向传播

                        loss_t = F.multilabel_soft_margin_loss(x11, label1) + F.multilabel_soft_margin_loss(x22,label1)

                        dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
                        if idx_test%10==0:
                            print('time:{}\t'
                              'Batch: [{:4d}/{:4d}]\t'
                              'Loss {:.4f})\t'.format(dt, idx_test, len(val_loader), loss_t.data.item()))

                        ind = torch.nonzero(label1)  # [10, 28] -> 非0元素的行列索引

                        for i in range(ind.shape[0]):  # 非0元素的个数

                            batch_index, la = ind[i]  # 帧索引，类别索引
                            save_path_hh = './val/'   # 保存地址

                            save_accu_map_folder = os.path.join(save_path_hh, str(save_index), img_name1[i][-18:-7])
                            if not os.path.exists(save_accu_map_folder):
                                os.makedirs(save_accu_map_folder)
                            save_accu_map_path = os.path.join(save_accu_map_folder, img_name1[i][-6:])
                            atts = (map1[i] + map2[i]) / 2          # 计算两幅图的平均值
                            atts[atts < 0] = 0
                            att = atts[la].cpu().data.numpy()       # 转为numpy数组
                            att = np.rint(att / (att.max() + 1e-8) * 255)   # 归一化到0-255
                            att = np.array(att, np.uint8)
                            att = cv2.resize(att, (220, 220))       # 修改分辨率

                            cv2.imwrite(save_accu_map_path[:-4] + '.png', att)   # 保存图片
                            heatmap = cv2.applyColorMap(att, cv2.COLORMAP_JET)   # 制作att伪彩色图像
                            img = cv2.imread(img_name1[i])
                            img = cv2.resize(img, (220, 220))
                            result = heatmap * 0.3 + img * 0.5      # 将原图和伪彩色图像重叠起来
                            cv2.imwrite(save_accu_map_path, result)      # 保存图像
                if not os.path.exists(os.path.join('./val/model/')):
                    os.makedirs(os.path.join('./val/model/'))
                savepath = os.path.join('./val/model/', str(save_index) + '.pth')
                torch.save(model.state_dict(), savepath)            # 保存现在的权重
                model.train()                                       # 改回训练模式

        if epoch %10 ==0 :                         # 如果现在的轮数=最终的轮数-1
            save_checkpoint(args,                                   # 保存检查点
                            {
                                'epoch': epoch,             # 当前轮数
                                'global_counter': global_counter,   # 优化器次数
                                'state_dict': model.state_dict(),   # 模型参数
                                'optimizer': optimizer.state_dict() # 优化器参数
                            }, is_best=False,
                            filename='%s_epoch_%d.pth' % (args.dataset, epoch))
        epoch += 1


if __name__ == '__main__':

    args = get_parser()                          # 获得命令行参数
    print('Running parameters:\n', args)            # 打印
    if not os.path.exists(args.checkpoint_dir):       # 确保快照文件夹存在
        os.makedirs(args.checkpoint_dir)
    save_index = 0                                  # 已经保存的次数
    train(args, save_index)

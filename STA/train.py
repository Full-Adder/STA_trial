import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import KLDLoss
from utils.args_config import get_parser
from utils.DataLoader import get_dataLoader
from utils.avgMeter import AverageMeter
from datetime import datetime
from utils.model_tool import get_model, save_checkpoint
from test import test
from torch.utils.tensorboard import SummaryWriter


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss2 = nn.BCEWithLogitsLoss().to(device)
    loss1 = KLDLoss().to(device)
    losses = AverageMeter()

    train_loader = get_dataLoader(Pic_path=args.Pic_path, train_mode="train", STA_mode=args.STA_mode,
                                  batch_size=args.batch_size, input_size=args.input_size,
                                  crop_size=args.crop_size)
    model, optimizer = get_model(args.STA_mode, args.lr, args.weight_decay)
    print(model)
    audiocls = torch.load('27001.pt')
    audiocls.cuda().eval()

    save_weight_fold = os.path.join(args.save_dir, args.STA_mode, './model_weight/')        # 权重保存地点
    best_pth = os.path.join(save_weight_fold, '%s_%s_model_bast.pth.tar' % (args.dataset_name, args.STA_mode))
    if os.path.exists(best_pth):
        print("-----> find pretrained model weight in", best_pth)
        state = torch.load(best_pth)
        total_epoch = state['epoch'] + 1
        if total_epoch >= args.epoch:
            print("-----> has trained", total_epoch + 1, 'in', best_pth)
            print("-----> your arg.epoch:", args.epoch)
            return
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print("-----> success load pretrained weight form ", best_pth)
        print("-----> continue to train!")
    else:
        total_epoch = 0

    writer = SummaryWriter(os.path.join(args.save_dir, args.STA_mode, './log/'))        # tensorboard保存地点
    val_re_save_path = os.path.join(args.save_dir, args.STA_mode, './val_re/')          # 验证集结果保存地点

    for epoch in range(total_epoch, args.epoch):
        model.train()
        losses.reset()

        for idx, data in enumerate(train_loader):  # 从train_loader中获取数据

            img_name, \
            img_bef, aud_bef, gt_bef, \
            img_now, aud_now, gt_now, \
            img_aft, aud_aft, gt_aft, \
            class_id, onehot_label = data

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

            if epoch == 0 and idx == 0:
                writer.add_graph(model, [img_bef, img_now, img_aft])

            p04, p03, p02, p14, p13, p12, p24, p23, p22 = \
                model(img_bef, img_now, img_aft, aud_bef, aud_now, aud_aft,
                      switch_bef, switch_now, switch_aft)

            loss_train = loss2(p04, gt_bef) + loss2(p14, gt_now) + loss2(p24, gt_aft) + \
                         loss2(p03, gt_bef) + loss2(p13, gt_now) + loss2(p23, gt_aft) + \
                         loss2(p02, gt_bef) + loss2(p12, gt_now) + loss2(p22, gt_aft) + \
                         loss1(F.sigmoid(p04), gt_bef) + loss1(F.sigmoid(p14), gt_now) + \
                         loss1(F.sigmoid(p24), gt_aft) + loss1(F.sigmoid(p03), gt_bef) + \
                         loss1(F.sigmoid(p13), gt_now) + loss1(F.sigmoid(p23), gt_aft) + \
                         loss1(F.sigmoid(p02), gt_bef) + loss1(F.sigmoid(p12), gt_now) + \
                         loss1(F.sigmoid(p22), gt_aft)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), img_now.size()[0])
            writer.add_scalars(args.dataset_name, {"train_loss": losses.val}, epoch * len(train_loader) + idx)

            if (idx + 1) % args.disp_interval == 0:
                dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
                print('time:{}\t'
                      'Epoch: [{:2d}][{:4d}/{:4d}]\t'
                      'LR: {:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.
                      format(dt, epoch + 1, (idx + 1), len(train_loader),
                             optimizer.param_groups[0]['lr'], loss=losses))

        if (epoch + 1) % args.val_Pepoch == 0:
            print("------------------------------val:start-----------------------------")
            test(model, args.Pic_path, args.H5_path, True, epoch, args.batch_size,
                 args.input_size, args.dataset_name, writer, val_re_save_path)
            print("------------------------------ val:end -----------------------------")

            if not os.path.exists(save_weight_fold):
                os.makedirs(save_weight_fold)
            save_path = os.path.join(save_weight_fold,
                                     "%s_%s_%03d" % (args.dataset_name, args.STA_mode, epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_path)  # 保存现在的权重
            print("weight has been saved in ", save_path)

            model.train()

        if (epoch + 1) == args.epoch:
            save_checkpoint({
                'epoch': epoch,  # 当前轮数
                'state_dict': model.state_dict(),  # 模型参数
                'optimizer': optimizer.state_dict()  # 优化器参数
            }, filename=os.path.join(save_weight_fold, '%s_%s_model_bast.pth.tar' % (args.dataset_name, args.STA_mode)))

    writer.close()


if __name__ == '__main__':

    args = get_parser()  # 获得命令行参数
    train(args)

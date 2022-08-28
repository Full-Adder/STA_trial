import os
import torch
import torch.nn.functional as F
from utils.args_config import get_parser
from utils.DataLoader import get_dataLoader
from utils.avgMeter import AverageMeter
from datetime import datetime
from utils.model_tool import get_model, save_checkpoint
from test import test
from torch.utils.tensorboard import SummaryWriter


def train(args):
    writer = SummaryWriter(args.SummaryWriter_dir)

    losses = AverageMeter()
    total_epoch = args.epoch

    train_loader = get_dataLoader(Pic_path=args.Pic_path, train_mode="val", STA_mode=args.STA_mode,
                                  batch_size=args.batch_size, input_size=args.input_size,
                                  crop_size=args.crop_size)
    model, optimizer = get_model(args.STA_mode, args.lr, args.weight_decay)

    for epoch in range(total_epoch):
        model.train()
        losses.reset()

        for idx, dat in enumerate(train_loader):  # 从train_loader中获取数据

            img_name1, img1, inda1, label1 = dat
            label1 = label1.cuda(non_blocking=True)
            img1 = img1.cuda(non_blocking=True)

            x11, x22, map1, map2 = model(img1)
            loss_train = F.multilabel_soft_margin_loss(x11, label1) + F.multilabel_soft_margin_loss(x22, label1)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), img1.size()[0])
            writer.add_scalars(args.dataset_name, {"train_loss": losses.avg}, epoch * len(train_loader) + idx)

            if (idx + 1) % args.disp_interval == 0:
                dt = datetime.now().strftime("%y-%m-%d %H:%M:%S")
                print('time:{}\t'
                      'Epoch: [{:2d}][{:4d}/{:4d}]\t'
                      'LR: {:.5f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    dt, epoch + 1, (idx + 1), len(train_loader),
                    optimizer.param_groups[0]['lr'], loss=losses))

        if (epoch + 1) % args.val_Pepoch == 0:
            print("------------------------------val:start-----------------------------")
            with torch.no_grad():
                test(model, args.Pic_path, True, epoch, args.batch_size, args.input_size, args.dataset_name)
            print("------------------------------ val:end -----------------------------")

            if not os.path.exists(os.path.join('./val/model/')):
                os.makedirs(os.path.join('./val/model/'))
            save_path = os.path.join('./val/model/', "%s_%03d" % (args.dataset_name, epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_path)  # 保存现在的权重
            print("weight has been saved in ", save_path)

            model.train()

        if (epoch + 1) == args.epoch:
            save_checkpoint({
                'epoch': epoch + 1,  # 当前轮数
                'state_dict': model.state_dict(),  # 模型参数
                'optimizer': optimizer.state_dict()  # 优化器参数
            },
                is_best=True, checkpoint_dir=args.checkpoint_dir,
                filename='%s_epoch_%d.pth' % (args.dataset_name, epoch + 1))

    writer.close()

if __name__ == '__main__':

    args = get_parser()  # 获得命令行参数
    print('Running parameters:\n', args)
    if not os.path.exists(args.SummaryWriter_dir):
        os.makedirs(args.SummaryWriter_dir)
    train(args)

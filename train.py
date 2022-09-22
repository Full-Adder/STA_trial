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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    losses = AverageMeter()

    train_loader = get_dataLoader(Pic_path=args.Pic_path, H5_path=args.H5_path, train_mode="train",
                                  STA_mode=args.STA_mode, batch_size=args.batch_size,
                                  input_size=args.input_size, crop_size=args.crop_size)
    model, optimizer = get_model(args.STA_mode, args.lr, args.weight_decay)

    save_weight_fold = os.path.join(args.save_dir, args.STA_mode, './model_weight/')  # 权重保存地点
    best_pth = os.path.join(save_weight_fold, '%s_%s_model_best.pth.tar' % (args.dataset_name, args.STA_mode))
    if os.path.exists(best_pth):
        print("-----> find pretrained model weight in", best_pth)
        state = torch.load(best_pth)
        total_epoch = state['epoch'] + 1
        print("-----> has trained", total_epoch + 1, 'in', best_pth)
        if total_epoch >= args.epoch:
            print("-----> your arg.epoch:", args.epoch)
            return
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print("-----> success load pretrained weight form ", best_pth)
        print("-----> continue to train!")
    else:
        total_epoch = 0

    writer = SummaryWriter(os.path.join(args.save_dir, args.STA_mode, './train_log/'))  # tensorboard保存地点
    val_re_save_path = os.path.join(args.save_dir, args.STA_mode, './val_re/')

    for epoch in range(total_epoch, args.epoch):
        model.train()
        losses.reset()

        for idx, dat in enumerate(train_loader):  # 从train_loader中获取数据

            if args.STA_mode == "S":
                img_name, img1, class_id, onehot_label = dat
                class_id = class_id.to(device)
                img1 = img1.to(device)

                if epoch == 0 and idx == 0:
                    writer.add_graph(model, img1)

                x11, x22, map1, map2 = model(img1)
                loss_train = F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)

            elif args.STA_mode == "SA":
                img_name, img1, aud1, class_id, onehot_label = dat
                img1 = img1.to(device)
                aud1 = aud1.to(device)
                class_id = class_id.to(device)

                if epoch == 0 and idx == 0:
                    writer.add_graph(model, [img1, aud1])

                x11, x22, map1, map2 = model(img1, aud1)
                loss_train = F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)

            elif args.STA_mode == "ST":
                img_name, img_bef, img1, img_aft, class_id, onehot_label = dat

                img_bef = img_bef.to(device)
                img1 = img1.to(device)
                img_aft = img_aft.to(device)
                class_id = class_id.to(device)

                if epoch == 0 and idx == 0:
                    writer.add_graph(model, [img_bef, img1, img_aft])

                x11, x1, x22, x2, x33, x3, map1, map2 = model(img_bef, img1, img_aft)
                loss_train = 0.4 * (F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)
                                    + F.cross_entropy(x33, class_id)) \
                             + 0.6 * (F.cross_entropy(x1, class_id) + F.cross_entropy(x2, class_id)
                                      + F.cross_entropy(x3, class_id))
            else:
                pass

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses.update(loss_train.data.item(), img1.size()[0])
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
            }, filename=os.path.join(save_weight_fold, '%s_%s_model_best.pth.tar' % (args.dataset_name, args.STA_mode)))

    writer.close()


if __name__ == '__main__':
    args = get_parser()  # 获得命令行参数
    train(args)

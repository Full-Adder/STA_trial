import os
import torch
import shutil  # 文件操作
import torch.optim as optim
from S.Smodel import SNetModel
from torch.utils.tensorboard import SummaryWriter


def get_model(STA_mode, lr=0.00005, weight_decay=0.0005):  # 获取 model
    assert STA_mode in ["S", "ST", "SA", "STA"], "STA_mode must be S/ST/SA/STA"
    if STA_mode == "S":
        model = SNetModel()  # 获得S
    else:
        model = SNetModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU
    model = torch.nn.DataParallel(model).cuda()
    model.to(device)  # 加载到GPU:0
    param_groups = model.module.get_parameter_groups()  # 获取 1作者定义的网络的权重 2作者定义的网罗的偏置 3resnext的权重 4resnext的偏置
    optimizer = optim.SGD([  # 随机梯度下降
        {'params': param_groups[0], 'lr': lr},  # 对于不同类型参数用不同的学习率
        {'params': param_groups[1], 'lr': 2 * lr},
        {'params': param_groups[2], 'lr': 10 * lr},
        {'params': param_groups[3], 'lr': 20 * lr}], momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return model, optimizer  # 返回模型和优化器


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)
    if is_best:  # 如果是最好的
        shutil.copyfile(save_path, os.path.join(checkpoint_dir, 'model_best.pth.tar'))  # 文件命名best复制到快照文件夹
    print("Congratulations! Your train code has been processed!")
    print("And your --epoch  --modul_weight  -- optim_weigh has been saved in ", save_path)


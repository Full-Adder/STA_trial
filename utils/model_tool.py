import os
import torch
import torch.onnx
import torch.optim as optim
from S.Smodel import SNetModel
from SA.SAmodel import SANetModel
from ST.STModel import STNetModel
from STA.STAModel import STANetModel
from utils.DataLoader import get_dataLoader
from torch.utils.tensorboard import SummaryWriter


def get_model(STA_mode, lr=0.00005, weight_decay=0.0005):  # 获取 model
    assert STA_mode in ["S", "ST", "SA", "STA"], "STA_mode must be S/ST/SA/STA"
    if STA_mode == "S":
        model = SNetModel()  # 获得S
    elif STA_mode == "SA":
        model = SANetModel()
    elif STA_mode == "ST":
        model = STNetModel()
    else:   # STA
        model = STANetModel()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 加载到GPU:0
    param_groups = model.get_parameter_groups()  # 获取 1作者定义的网络的权重 2作者定义的网罗的偏置 3resnext的权重 4resnext的偏置
    optimizer = optim.SGD([  # 随机梯度下降
        {'params': param_groups[0], 'lr': lr},  # 对于不同类型参数用不同的学习率
        {'params': param_groups[1], 'lr': 2 * lr},
        {'params': param_groups[2], 'lr': 10 * lr},
        {'params': param_groups[3], 'lr': 20 * lr}], momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return model, optimizer  # 返回模型和优化器


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print("Congratulations! Your train code has been processed!")
    print("And your --epoch  --modul_weight  -- optim_weigh has been saved in ", filename)


def save_model_to_oxxn(STA_mode):
    model, _ = get_model(STA_mode)
    state = torch.load("../Result/S/model_weight/AVE_S_030.pth")
    model.load_state_dict(state)
    model.to('cpu')
    dataload = get_dataLoader(STA_mode=STA_mode, batch_size=1, train_mode="val")
    _, x, _, _ = next(iter(dataload))
    print(x,x.shape)
    torch.onnx.export(model, x, "../S_model.onnx", export_params=True, verbose=True)


if __name__ == '__main__':
    save_model_to_oxxn("S")
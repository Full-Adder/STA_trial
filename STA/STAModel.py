import torch.nn as nn
from Soundmodel import SoundNet
import torch
import torch.nn.functional as F
affine_par = True
from ConvGRU import ConvGRUCell
import os
from torch.nn import init

model_urls = {'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class STANetModel(nn.Module):
    def  __init__(self):
        super(STANetModel, self).__init__()
        self.extra_audio_d = nn.Linear(8192, 2048)
        self.extra_convs = nn.Conv2d(2048, 32, kernel_size=1)
        self.extra_convsd1 = nn.Conv2d(1024, 32, kernel_size=1)
        self.extra_convsd0 = nn.Conv2d(512, 32, kernel_size=1)
        all_channel = 32
        self.extra_conv_fusion = nn.Conv2d(all_channel*3, all_channel, kernel_size=3, padding=1, bias=True)
        self.extra_ConvGRU = ConvGRUCell(all_channel, all_channel, kernel_size=1)
        self.extra_gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.extra_gate_s = nn.Sigmoid()
        self.extra_gates = nn.Conv2d(all_channel, 1, kernel_size = 1, bias = False)
        self.extra_gates_s = nn.Sigmoid()
        self.extra_gateav = nn.Sequential(nn.Conv2d(all_channel*4, 32, 1),nn.ReLU(True),nn.Conv2d(32, 1, 1))

        self.extra_gateav_s = nn.Sigmoid()
        self.extra_projf = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projg = nn.Conv2d(in_channels=all_channel, out_channels=all_channel // 2, kernel_size=1)
        self.extra_projh = nn.Conv2d(in_channels=all_channel, out_channels=all_channel, kernel_size=1)
        self.extra_lineara = nn.Linear(8192, 128)
        self.extra_linearv = nn.Linear(2048, 128)
        self.extra_refineST = nn.Sequential(
            nn.Conv3d(32, 32, (3, 1, 1), padding=(0, 0, 0)),
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(32, 32, (1, 3, 3), padding=(0, 1, 1)))
        self.extra_s = nn.Sigmoid()
        self.extra_refineSA = nn.Sequential(nn.Conv2d(512, all_channel, kernel_size=1), nn.Conv2d(all_channel, 1, 1), nn.Sigmoid())
        self.extra_convv = nn.Conv2d(2048, 128, kernel_size=1)
        self.extra_conv_stafusion = nn.Conv2d(all_channel*3, all_channel, kernel_size=3, padding=1, bias= True)
        self.extra_refine1 = nn.Sequential(nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=True))
        self.extra_refine0 = nn.Sequential(nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias=True))
        self.extra_out1 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=True)
        self.extra_out0 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=True)
        self.att_dir = None


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        Amodel = SoundNet()
        checkpoint = torch.load('./SA/vggsound_netvlad.pth.tar')
        Amodel.load_state_dict(checkpoint['model_state_dict'])
        Amodel = list(Amodel.audnet.children())
        self.audio_model = nn.Sequential(*Amodel[:9])

        # net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        net = torch.hub.load('./tmp/facebookresearch_WSL-Images_main', 'resnext101_32x8d_wsl', source='local')
        net = list(net.children())
        self.features0 = nn.Sequential(*net[:6])
        self.features1 = nn.Sequential(*net[6])
        self.features2 = nn.Sequential(*net[7])

    def forward(self, img0, img1, img2, aud0, aud1, aud2):

        batch_num  = img0.size()[0]

        x0s0 = self.features0(img0) # 512,45,45
        x0s1 = self.features1(x0s0) # 1024,23,23
        x0s = self.features2(x0s1) # 2048,12,12
        x0s0 = self.extra_convsd0(x0s0)
        x0s1 = self.extra_convsd1(x0s1)

        x0 = self.extra_convs(x0s) # 7,32,12,12
        x0ss = F.avg_pool2d(x0, kernel_size=(x0.size(2), x0.size(3)), padding=0) # 1,32,1,1
        x0ss = x0ss.view(-1, 32) # 1,32
        a0 = self.audio_model(aud0.unsqueeze(1))  # [7, 8192]
        av0 = self.AVfusion(a0, x0s)

        x1s0 = self.features0(img1) # 512,45,45
        x1s1 = self.features1(x1s0) # 1024,23,23
        x1s = self.features2(x1s1) # 2048,12,12
        x1s0 = self.extra_convsd0(x1s0)
        x1s1 = self.extra_convsd1(x1s1)

        x1 = self.extra_convs(x1s) # 1,32,8,8
        x1ss = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1ss = x1ss.view(-1, 32)
        a1 = self.audio_model(aud1.unsqueeze(1))  # [7, 8192]
        av1 = self.AVfusion(a1, x1s)

        x2s0 = self.features0(img2) # 512,45,45
        x2s1 = self.features1(x2s0) # 1024,23,23
        x2s = self.features2(x2s1) # 2048,12,12
        x2s0 = self.extra_convsd0(x2s0)
        x2s1 = self.extra_convsd1(x2s1)

        x2 = self.extra_convs(x2s) # 1,32,8,8
        x2ss = F.avg_pool2d(x2,kernel_size=(x2.size(2),x2.size(3)),padding=0)
        x2ss = x2ss.view(-1, 32) # 2,32
        a2 = self.audio_model(aud2.unsqueeze(1))  # [7, 8192]
        av2 = self.AVfusion(a2, x2s)

        incat0 = torch.cat((x2.unsqueeze(1), x0.unsqueeze(1), x1.unsqueeze(1)), 1).view(batch_num,32,3,x0.size(2), x0.size(3))
        incat1 = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), 1).view(batch_num,32,3,x1.size(2), x1.size(3))
        incat2 = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x0.unsqueeze(1)), 1).view(batch_num,32,3,x2.size(2), x2.size(3))

        x00 = self.extra_conv_fusion(torch.cat((F.relu(x0+self.self_attention(x0)), F.relu(x0+x0*self.extra_s(self.extra_refineST(incat0).squeeze(2))), F.relu(x0+x0*av0)), 1))
        x00 = self.extra_ConvGRU(x00, x0)
        x00refine1 = F.relu(self.extra_refine1(torch.cat((F.upsample(x00, size=x0s1.size()[2:], mode='bilinear'),x0s1),1))+x0s1,True)
        x00refine1 = F.upsample(x00refine1, size=x0s0.size()[2:], mode='bilinear')
        x00refine0 = F.relu(self.extra_refine0(torch.cat((x00refine1,x0s0),1))+x00refine1,True)
        out0_1 = F.upsample(self.extra_out1(x00refine1), size=img0.size()[2:], mode='bilinear')
        out0_0 = F.upsample(self.extra_out0(x00refine0), size=img0.size()[2:], mode='bilinear')

        x11 = self.extra_conv_fusion(torch.cat((F.relu(x1+self.self_attention(x1)), F.relu(x1+x1*self.extra_s(self.extra_refineST(incat1).squeeze(2))), F.relu(x1+x1*av1)), 1))
        x11 = self.extra_ConvGRU(x11, x1)
        x11refine1 = F.relu(self.extra_refine1(torch.cat((F.upsample(x11, size=x1s1.size()[2:], mode='bilinear'),x1s1),1))+x1s1,True)
        x11refine1 = F.upsample(x11refine1, size=x1s0.size()[2:], mode='bilinear')
        x11refine0 = F.relu(self.extra_refine0(torch.cat((x11refine1,x1s0),1))+x11refine1,True)
        out1_0 = F.upsample(self.extra_out0(x11refine0), size=img0.size()[2:], mode='bilinear')
        out1_1 = F.upsample(self.extra_out1(x11refine1), size=img0.size()[2:], mode='bilinear')

        x22 = self.extra_conv_fusion(torch.cat((F.relu(x2+self.self_attention(x2)), F.relu(x2+x2*self.extra_s(self.extra_refineST(incat2).squeeze(2))), F.relu(x2+x2*av2)), 1))
        x22 = self.extra_ConvGRU(x22, x2)
        x22refine1 = F.relu(self.extra_refine1(torch.cat((F.upsample(x22, size=x2s1.size()[2:], mode='bilinear'),x2s1),1))+x2s1,True)
        x22refine1 = F.upsample(x22refine1, size=x2s0.size()[2:], mode='bilinear')
        x22refine0 = F.relu(self.extra_refine0(torch.cat((x22refine1,x2s0),1))+x22refine1,True)
        out2_0 = F.upsample(self.extra_out0(x22refine0), size=img0.size()[2:], mode='bilinear')
        out2_1 = F.upsample(self.extra_out1(x22refine1), size=img0.size()[2:], mode='bilinear')

        return out0_1,out0_0, out1_1,out1_0, out2_0,out2_1

    def self_attention(self, x):
        m_batchsize, C, width, height = x.size()  # 8,32,8,8
        f = self.extra_projf(x).view(m_batchsize, -1, width * height)
        g = self.extra_projg(x).view(m_batchsize, -1, width * height)
        h = self.extra_projh(x).view(m_batchsize, -1, width * height)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = F.softmax(attention, dim=1)  # 8,1024,1024

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H
        self_mask = self.extra_gates(self_attetion)  # [1, 1, 8, 8]
        self_mask = self.extra_gates_s(self_mask)
        out = self_mask * x
        return out

    def AVfusion(self, audio, visual):
        bs, C, H, W = visual.shape
        visuals = self.extra_convv(visual)
        a_fea = self.extra_lineara(audio)
        a_fea = a_fea.view(bs, -1).unsqueeze(2) # [7, 1, 1024]

        video_t= F.avg_pool2d(visual,kernel_size=(visual.size(2),visual.size(3)),padding=0) # [7, 2048, 1, 1]
        video_t = video_t.view(bs, -1) #[7, 2048]
        v_fea = self.extra_linearv(video_t).unsqueeze(1) # [[7, 1, 1024]]

        att_wei = torch.bmm(a_fea, v_fea) # [bs*10, 1, 49]
        att_wei = F.softmax(att_wei, dim=-1) # 2,128,128
        att_v_fea = torch.bmm(att_wei, visuals.view(bs, 128, H*W))
        att_v_fea = att_v_fea.view(bs, 128, H, W)
        self_mask = self.extra_gateav(att_v_fea)  # [1, 1, 8, 8]
        self_mask = self.extra_gateav_s(self_mask)
        return self_mask

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups
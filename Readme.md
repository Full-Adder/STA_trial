# STA实验报告

[1] [作者的代码:guotaowang/STANet (github.com)](https://github.com/guotaowang/STANet)

[2] [修改后本实验代码:Full-Adder/STA_trial (github.com)](https://github.com/Full-Adder/STA_trial)

## 实验介绍

此次实验是论文《[From Semantic Categories to Fixations: A Novel Weakly-supervised Visual-auditory Saliency Detection Approach ](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_From_Semantic_Categories_to_Fixations_A_Novel_Weakly-Supervised_Visual-Auditory_Saliency_CVPR_2021_paper.pdf)》的代码实现。

针对目前只有少数带有真实注视点视听序列的问题，论文中提出了一种弱监督的方式，使用一种遵循粗到细(coarse-to-fine)策略的选择性类激活映射(SCAM)，仅通过视频类别标签来获得空间S-时间T-音频A环境中最显著的区域。并且预测结果后续可以作为伪-真实数据训练一个新的STA网络。

实验总体可以分为三步：

1. *SCAM training*

> **Course** ：分别训练 $S_{coarse}$, $SA_{coarse} $, $ {ST}_{coarse} $，输入大小256确保物体定位准确。
>
> **Fine** ：分别再训练  $S_{fine}$, $ {SA}_{fine} $, $ {ST}_{fine} $，输入大小356确保区域定位准确。

2. *pseudoGT generation*

> 为了便于处理矩阵数据时的展示，在流畅的帧间进行粗糙定位和伪-GT数据后处理时使用 $Matlab2016b$ 进行处理。

3. *STA training*

> 使用带有生成伪-GT的AVE视频帧来训练STA模型。

项目目录如下：

<img src="Readme.assets/image-20220921234526308.png" alt="image-20220921234526308" style="zoom:50%;" />

## AVE数据集介绍

AVE数据集结构如下：

```
      │  Annotations.txt
      │  ReadMe.txt
      │  testSet.txt
      │  trainSet.txt
      │  valSet.txt
      └─AVE
        ---1_cCGK4M.mp4
        --12UOziMF0.mp4
        --5zANFBYzQ.mp4
        --9O4XZOge4.mp4
        --bSurT-1Ak.mp4
```

AVE 数据集包含4096个时长为10s的MP4视频，每个视频被唯一地标注了一个视频种类标签，数据集共被标注了28个视频分类标签。视频的类别等信息存储在 $Annotations.txt$, $test.txt$, $train.txt$ 中，存储数据如下：


| Category        | VideoID     | Quality | StartTime | EndTime |
| --------------- | ----------- | ------- | --------- | ------- |
| Church bell     | RUhOCu3LNXM | good    | 0         | 10      |
| Church bell     | MH3m4AwEcRY | good    | 6         | 8       |
| Acoustic guitar | -AFx6goDrOw | good    | 0         | 10      |
| ...             | ...         | ...     | ...       | ...     |

从左到右分别为视频的类别标签，视频的名称(ID)，视频对分类结果的质量，和类别标签在视频中开始和结束的时间(s)。其中，训练集包含3339个MP4文件，占比81.5%，验证集和测试集分别包含402个MP4文件，占比9.8%。

## 实验过程

### 数据准备

> 解压原始数据集后，修改文件夹，所有MP4视频存放于Video_path目录下，运行`AVE_modify.py`, 即可获得存放在`Pic_path`、`Audio_path`、`H5_path` 中 `.jpg` 和 `.h5` 文件.

在训练前需要将数据集MP4文件转化成每帧的.jpg、音频的.wav/.h5格式文件，本实验提取了视频对应的 $StartTime-EndTime$  中每秒的视频帧和每秒的音频特征（例如在4-7s的视频中提取4.5s的视频帧和4-5s的音频特征作为4-5s的空间和音频特征）。

#### 读取$train.txt/test.txt/val.txt$文件

> 数据集处理完毕后，从`txt`中获取文件名，向`dataset`输送。

```python
def readDataTxt(DataSet_path, mode):
    txt_path = os.path.join(DataSet_path, txt_name[mode])
    with open(txt_path, 'r', encoding="utf-8") as f:
        if mode == "all":
            f.readline()
        dataSet = []
        for data in f.readlines():
            data = data.strip().split('&')
            data_list = [data[1], category_id[data[0]], int(data[3]), int(data[4])]
            dataSet.append(data_list)

    return dataSet
# 返回文件名，对应类别，开始时间，结束时间 的列表
```

#### 数据集类

```python
class AVEDataset(Dataset):  # 数据集类
    def __init__(self, pic_dir, h5_dir, mode, transform=None, STA_mode="S"):
        assert mode in list(dft.get_txtList().keys()), "mode must be train/test/val"
        assert STA_mode in ["S", "ST", "SA", "STA"], "STA_mode must be S/SA/ST/STA"
        self.pic_dir = pic_dir
        self.h5_dir = h5_dir
        self.mode = mode
        self.transform = transform
        self.STA_mode = STA_mode
        self.class_dir = dft.get_category_to_key()
        data_folder_list = dft.readDataTxt(os.path.join(self.pic_dir, "../"), mode)
        self.data_list = []
        for idata in data_folder_list:
            if self.STA_mode == "S" or self.STA_mode == "SA":
                for idx in range(idata[-2], idata[-1]):
                    self.data_list.append([os.path.join(idata[0], "{:0>2d}".format(idx)), idata[1]])
            else:
                for idx in range(idata[-2] + 1, idata[-1] - 1):
                    self.data_list.append([os.path.join(idata[0], "{:0>2d}".format(idx)), idata[1]])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        video_name, id = data[0][:11], int(data[0][-2:])

        img_path = os.path.join(self.pic_dir, video_name, "%02d" % id + ".jpg")  # 图片绝对地址
        image = Image.open(img_path).convert('RGB')  # 打开图片，并转化为RGB格式
        image = self.transform(image)  # 将图片转化为tensor

        class_id = int(data[1])  # 获得图像标签
        onehot_label = np.zeros(len(self.class_dir), dtype=np.float32)
        onehot_label[class_id] = 1  # one-hot编码

        if self.STA_mode == "S":
            return data[0], image, class_id, onehot_label  # 返回 图片地址 图片tensor 标签 onehot标签

        elif self.STA_mode == "SA":
            h5_path = os.path.join(self.h5_dir, data[0] + ".h5")
            with h5py.File(h5_path, 'r') as hf:
                audio_features = np.float32(hf['dataset'][:])  # 5,128
            audio = torch.from_numpy(audio_features).float()
            return data[0], image, audio, class_id, onehot_label  # 返回 图片地址 图片tensor 标签 onehot标签

        else:
            img_bef_path = os.path.join(self.pic_dir, video_name, "%02d" % (id - 1) + ".jpg")  # 前一张图片的地址
            img_aft_path = os.path.join(self.pic_dir, video_name, "%02d" % (id + 1) + ".jpg")  # 后一张图片的地址
            image_bef = Image.open(img_bef_path).convert('RGB')
            image_aft = Image.open(img_aft_path).convert('RGB')
            image_bef = self.transform(image_bef)
            image_aft = self.transform(image_aft)

            if self.STA_mode == "ST":
                return data[0], image_bef, image, image_aft, class_id, onehot_label

            else:  # "STA"
                pass

```

#### 数据加载器

```python
def get_dataLoader(Pic_path=r"../AVE_Dataset/Video", H5_path=r"../AVE_Dataset/H5", train_mode="train",
                   STA_mode="S", batch_size=8, input_size=300, crop_size=256):
    mean_vals = [0.485, 0.456, 0.406]  # 数据均值
    std_vals = [0.229, 0.224, 0.225]  # 数据标准差

    tsfm_train = transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.RandomCrop((crop_size, crop_size)),
                                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals),
                                     ])

    tsfm_test = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])

    if train_mode == "train":
        img_train = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, mode="train", transform=tsfm_train, STA_mode=STA_mode)
        train_loader = DataLoader(img_train, batch_size=batch_size, shuffle=True, drop_last=True)
        print("dataSet.len:", len(img_train), "\t dataLoader.len:", len(train_loader), 'batch_size:', batch_size)
        return train_loader
    elif train_mode == "test":
        img_test = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, mode="test", transform=tsfm_test, STA_mode=STA_mode)
        test_loader = DataLoader(img_test, batch_size=batch_size, shuffle=False, drop_last=True)
        print("dataSet.len:", len(img_test), "\t dataLoader.len:", len(test_loader), 'batch_size:', batch_size)
        return test_loader
    elif train_mode == "val":
        img_val = AVEDataset(pic_dir=Pic_path, h5_dir=H5_path, mode="val", transform=tsfm_test, STA_mode=STA_mode)
        val_loader = DataLoader(img_val, batch_size=batch_size, shuffle=False, drop_last=True)
        print("dataSet.len:", len(val_loader), "\t dataLoader.len:", len(val_loader), 'batch_size:', batch_size)
        return val_loader
```

### S model

![image-20220831151542976](Readme.assets/S_model.png)


| x_name      | later_name                   | detail                                    | output_size         |
| ----------- | ---------------------------- | ----------------------------------------- | ------------------- |
| x           | input                        | none                                      | [1, 3, 256, 256]    |
| x1          | features(x)                  | nn.Sequential(*ResNetX[:7])               | [1, 1024, 19, 19]   |
| x11         | extra_convs(x1)              | nn.Conv2d(1024, 28, 1)                    | [1, 28, 19 ,19]     |
| x_att       | self_attention               |                                           |                     |
| _f          | extra_projf(x11)             | Conv2d(28, 14, 1).view(b, -1, w* h)       | [1, 14, 19*19]      |
| _g          | extra_projg(x11)             | Conv2d(28, 14, 1).view(b, -1, w* h)       | [1, 14, 19*19]      |
| _h          | extra_projh(x11)             | Conv2d(28, 28, 1).view(b, -1, w* h)       | [1, 28, 19*19]      |
| _atte       | bmm-矩阵相乘                 | softmax(tor.bmm(f[0,2,1],g))              | [1, 19\*19, 19\*19] |
| _self_atte  | 相乘后展开                   | bmm(h, atte).view(b, c, w,h)              | [1，28, 19, 19]     |
| _self_mask  | extra_gate(self_atte)        | sogmoid(Conv2d(28, 1, 1))                 | [1, 1, 19, 19]      |
|             |                              | self_mask * x11                           | [1, 28, 19, 19]     |
| x2          | extra_conv_fusion(x11,x_att) | Conv2d(56, 28, 1, True)                   | [1, 28, 19, 19]     |
| x22         | extra_ConvGRU(x2,x11)        |                                           |                     |
| _update     | tor.cat(x2,x11)后            | sigmoid(Conv2d(56, 28,1))                 | [1, 28, 19, 19]     |
| _reset      | tor.cat(x2,x11)后            | sigmoid(Conv2d(56, 28,1))                 | [1, 28, 19, 19]     |
| _out_inputs | cat([x11, x_att * reset])    | tanh(Conv2d(56, 28,1))                    | [1, 28, 19, 19]     |
|             | return                       | x_att* (1 - update) + out_inputs * update | [1, 28, 19, 19]     |
| map_1       |                              | x11.clone()                               |                     |
| x1ss        |                              | avg_pool2d(x11, 19, 19).view(-1, 28)      | [1, 28]             |
| map_2       |                              | x22.clone()                               |                     |
| x2ss        |                              | avg_pool2d(x22, 19, 19).view(-1, 28)      | [1, 28]             |
|             | return                       | x1ss, x2ss,map1, map2                     |                     |

（以batch_size = 1为例）

对应图中：

`F.softmax(x1ss, dim=1).data.squeeze()`后获得对应28个类别的预测概率，最大值即为预测的对应类别。

$atts = (map1[i] + map2[i]) / 2 $ 作为CAM结果。

##### loss

```python
# loss_train = F.multilabel_soft_margin_loss(x1ss, label1) +
# 			F.multilabel_soft_margin_loss(x2ss, label1)
```

~~$MultiLabelSoftMarginLoss$ 针对多分类，且每个样本只能属于一个类的情形~~

~~$\operatorname{MultiLabelSoftMarginLoss}(x, y)=-\frac{1}{C} * \sum_{i} y[i] * \log \left((1+\exp (-x[i]))^{-1}\right)+(1-y[i]) * \log \left(\frac{\exp (-x[i])}{1+\exp (-x[i])}\right) $~~

~~相当于对$min-batch$个多个交叉熵损失求平均值。~~

```python
loss_train = F.cross_entropy(x11, class_id)+F.cross_entropy(x22, class_id)
```
修改为使用[交叉熵损失](https://www.modb.pro/db/176110)

### SA model

![SA_model](Readme.assets/SA_model.png)


| x_name      | later_name                                                                            | detail                                            | output_size      |
| ----------- | ------------------------------------------------------------------------------------- | ------------------------------------------------- | :--------------- |
| aud1        |                                                                                       | input_audio                                       | [1, 81, 199]     |
| _aud1       | aud1.unsqueeze(1)                                                                     | 在第一维插入一个维度                              | [1, 1, 81, 199]  |
| a1          | audio_model(_aud1)                                                                    | nn.Sequential(*ResNet[:9])                        | [1, 8192]        |
| a1          | extra_audio_d(a1)                                                                     | nn.Linear(8192, 2048)                             | [1, 2048]        |
|             | return                                                                                | a1.unsqueeze(2).unsqueeze(2)                      | [1, 2048, 1, 1]  |
| Aup         | Aup(a1)                                                                               |                                                   |                  |
|             | [nn.ConvTranspose2d简介](https://blog.csdn.net/qq_27261889/article/details/86304061/) | nn.ConvTranspose2d(2048,2048,3,1,0) nn.ReLU(True) | [1, 2048, 3, 3]  |
|             |                                                                                       | nn.ConvTranspose2d(2048,2048,3,1,0) nn.ReLU(True) | [1, 2048, 4, 4]  |
|             | return                                                                                | nn.ConvTranspose2d(2048,2048,3,1,0)               | [1, 2048, 5, 5]  |
| convs_Aup   | extra_convs(Aup)                                                                      |                                                   |                  |
|             |                                                                                       | nn.Conv2d(2048, 28, 1)                            | [1, 28, 5, 5]    |
|             | return                                                                                | nn.Conv2d(28, 1, 1) nn.Sigmoid()                  | [1, 1, 5, 5]     |
| x           |                                                                                       | input_pic                                         | [1, 3, 256, 256] |
| x1          | features(x)                                                                           | nn.Sequential(*ResNetX[:8])                       | [1, 2048, 8, 8]  |
| x1          | extra_video_d(x1)                                                                     |                                                   |                  |
|             |                                                                                       | nn.Conv2d(2048, 2048,3, 1), nn.ReLU(True)         | [1, 2048, 7, 7]  |
|             |                                                                                       | nn.Conv2d(2048, 2048,3, 1), nn.ReLU(True)         | [1, 2048, 6, 6]  |
|             |                                                                                       | nn.Conv2d(2048, 2048,3, 1), nn.ReLU(True)         | [1, 2048, 5, 5]  |
|             | return                                                                                | nn.Conv2d(2048, 28, 1)                            | [1, 28, 5, 5]    |
| x_att       | self_attention                                                                        |                                                   |                  |
| _f          | extra_projf(x1)                                                                       | Conv2d(28, 14, 1).view(b, -1, w* h)               | [1, 14, 5*5]     |
| _g          | extra_projg(x1)                                                                       | Conv2d(28, 14, 1).view(b, -1, w* h)               | [1, 14, 5*5]     |
| _h          | extra_projh(x1)                                                                       | Conv2d(28, 28, 1).view(b, -1, w* h)               | [1, 28, 5*5]     |
| _atte       | softmax(tor.bmm(f[0,2,1],g))                                                          | bmm-矩阵相乘                                      | [1, 5\*5, 5\*5]  |
| _self_atte  | bmm(h,atte).view(b,c,w,h)                                                             | 相乘后展开                                        | [1，28, 5, 5]    |
| _self_mask  | extra_gate(self_atte)                                                                 | sogmoid(Conv2d(28,1,1))                           | [1, 1, 5, 5]     |
|             | return                                                                                | self_mask * x11                                   | [1, 28, 5, 5]    |
| xxatt       |                                                                                       | F.relu(x1 + x_att)                                | [1, 28, 5, 5]    |
| xxApu       |                                                                                       | F.relu(x1 + x1 * convs_Aup))                      | [1, 28, 5, 5]    |
| xxattApu    |                                                                                       | torch.cat(xxatt,xxApu,1)                          | [1, 56, 5, 5]    |
| x2          | extra_conv_fusion(xxattApu)                                                           | Conv2d(56,28,1,True)                              | [1, 28, 5, 5]    |
| x22         | extra_ConvGRU(x2,x1)                                                                  |                                                   |                  |
| _update     | tor.cat(x2,x1,1)后                                                                    | sigmoid(Conv2d(56,28,1))                          | [1, 28, 5, 5]    |
| _reset      | tor.cat(x2,x1,1)后                                                                    | sigmoid(Conv2d(56,28,1))                          | [1, 28, 5, 5]    |
| _out_inputs | cat([x2, x1 * reset])                                                                 | tanh(Conv2d(56,28,1))                             | [1, 28, 5, 5]    |
|             |                                                                                       | x1*(1-update)+out_inputs\*update                  | [1, 28, 5, 5]    |
| map_1       | x1.clone()                                                                            |                                                   |                  |
| x1ss        |                                                                                       | avg_pool2d(x1,5,5).view(-1,28)                    | [1, 28]          |
| map_2       | x2.clone()                                                                            |                                                   |                  |
| x2ss        |                                                                                       | avg_pool2d(x22,5,5).view(-1,28)                   | [1, 28]          |
|             | return                                                                                | x1ss, x2ss,map1, map2                             |                  |

（以batch_size = 1为例）

> 表中对应函数的默认位置参数如下：
>
> nn.Conv2d(input_size,output_size,k_size,bias)
>
> nn.ConvTranspose2d(input_size,output_size,k_size,strip,padding)

同理，对应图中：

`F.softmax(x1ss, dim=1).data.squeeze()`后获得对应28个类别的预测概率，最大值即为预测的对应类别。

$atts = (map1[i] + map2[i]) / 2$  作为CAM结果。

##### loss

SA的loss与S模型的loss一样：

```python
loss_train = F.cross_entropy(x11, class_id)+F.cross_entropy(x22, class_id)
```

### 模型训练与测试

#### train.py

```python
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
                class_id.to(device)
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

```

#### test.py

```python
def test(model, Pic_path, H5_path, is_val, save_index, batch_size,
         input_size, dataset_name, Summary_Writer, test_re_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    if is_val:
        save_path_hh = os.path.join(test_re_dir, str(save_index))
    else:
        save_path_hh = test_re_dir

    val_mode = "val" if is_val else "test"
    test_loader = get_dataLoader(Pic_path=Pic_path, H5_path=H5_path, train_mode=val_mode, STA_mode=args.STA_mode,
                                 batch_size=batch_size, input_size=input_size)  # 获取测试集
    # !!
    for idx_test, dat_test in enumerate(test_loader):
        with torch.no_grad():
            if args.STA_mode == "S":
                img_name, img1, class_id, onehot_label = dat_test
                img1 = img1.to(device)
                class_id = class_id.to(device)

                x11, x22, map1, map2 = model(img1)
                loss_t = F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)

            elif args.STA_mode == "SA":
                img_name, img1, aud1, class_id, onehot_label = dat_test
                img1 = img1.to(device)
                aud1 = aud1.to(device)
                class_id = class_id.to(device)

                x11, x22, map1, map2 = model(img1, aud1)
                loss_t = F.cross_entropy(x11, class_id) + F.cross_entropy(x22, class_id)

            elif args.STA_mode == "ST":
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
            else:
                pass

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
            probs, index_of_pic = h_x.sort(1, True)  # 1行排序
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
        test_epoch = input(r"please input your test weight of your mode! Maybe is 10/20/.../best: ")
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
```



## 实验结果

### S model

训练轮数：30

以下是训练损失和验证集损失：

![image-20220921232432121](Readme.assets/image-20220921232432121.png)



分别在30个轮次的训练上跑测试集损失。可见第11轮的测试集损失最小。

![image-20220921231649504](Readme.assets/image-20220921231649504.png)



以下是30轮（most）和11轮（test_loss_best）的对比：

![image-20220921231555996](Readme.assets/image-20220921231555996.png)

但是，从个人观察的结果来看，30轮结果相比于11轮显著性注视点更加集中，效果更好。

|                       |                 epoch = 30                 |                 epoch = 11                 |
| :-------------------: | :----------------------------------------: | :----------------------------------------: |
|       Chainsaw        | ![01](Readme.assets/01-16637742440853.jpg) | ![01](Readme.assets/01-16637742163162.jpg) |
| Race car, auto racing |        ![00](Readme.assets/00.jpg)         | ![00](Readme.assets/00-16637743691694.jpg) |
|     Frying (food)     | ![06](Readme.assets/06-16637744732186.jpg) | ![06](Readme.assets/06-16637744514275.jpg) |

以下是test_result (epoch=30)的展示:

|    label    |            right            |                            wrong                             |         wrong_result          |
| :---------: | :-------------------------: | :----------------------------------------------------------: | :---------------------------: |
| Church bell | ![01](Readme.assets/01.jpg) |  ![01(wrong_02_Bark)](Readme.assets/01(wrong_02_Bark).jpg)   |             Bark              |
|     bus     | ![03](Readme.assets/03.jpg) | ![01(wrong_04_Race car, auto racing)](Readme.assets/01(wrong_04_Race car, auto racing).jpg) |     Race car, auto racing     |
|    Flute    | ![06](Readme.assets/06.jpg) | ![07(wrong_01_Male speech, man speaking)](Readme.assets/07(wrong_01_Male speech, man speaking).jpg) |   Male speech, man speaking   |
|    horse    | ![02](Readme.assets/02.jpg) | ![03(wrong_03_Fixed-wing aircraft, airplane)](Readme.assets/03(wrong_03_Fixed-wing aircraft, airplane).jpg) | Fixed-wing aircraft, airplane |

第30轮的验证损失反而增高了，从个人理解应该是由于视频种类有许多相近的地方，例如有多种乐器分类的画面都是音乐家在演奏，仅从低分辨率的图像很难将各种乐器准确的分辨出来，并且演奏的画面又与说话的视频种类类似，这很大程度的导致了较大的损失。随着训练的进行，特征的区域更加精确，范围缩小，各分类的概率相差更大，使分类结果错误更多。

### SA model

训练轮数：10

训练时的验证集损失：

![image-20220921235102913](Readme.assets/image-20220921235102913.png)



### ST model

训练轮数：50轮

训练损失（蓝色）和验证集损失（红色）

![image-20220921225215974](Readme.assets/image-20220921225215974.png)

在








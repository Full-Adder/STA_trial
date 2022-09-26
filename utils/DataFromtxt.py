import os
from utils.args_config import get_parser

txt_name = {"train": "trainSet.txt", "test": "testSet.txt", "val": "valSet.txt", "all": "Annotations.txt"}
id_category = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane', 'Race car, auto racing', 'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar', 'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw', 'Cat', 'Horse', 'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin']
category_id = \
    {'Church bell': 0,                      # 教堂的钟
     'Male speech, man speaking': 1,        # 男人说话
     'Bark': 2,                             # 犬吠
     'Fixed-wing aircraft, airplane': 3,    # 固定翼飞机
     'Race car, auto racing': 4,            # 赛车
     'Female speech, woman speaking': 5,    # 女人说话
     'Helicopter': 6,               # 直升机
     'Violin, fiddle': 7,           # 小提琴
     'Flute': 8,                    # 长笛
     'Ukulele': 9,                  # 尤克里里
     'Frying (food)': 10,           # 油炸(食物)
     'Truck': 11,                   # 卡车
     'Shofar': 12,                  # 羊角号
     'Motorcycle': 13,              # 摩托车
     'Acoustic guitar': 14,         # 原声吉他
     'Train horn': 15,              # 火车鸣叫
     'Clock': 16,                   # 闹钟
     'Banjo': 17,                   # 班卓琴
     'Goat': 18,                    # 山羊
     'Baby cry, infant cry': 19,    # 婴儿啼哭
     'Bus': 20,                     # 公交车
     'Chainsaw': 21,                # 电锯
     'Cat': 22,                     # 猫
     'Horse': 23,                   # 马
     'Toilet flush': 24,            # 马桶冲水
     'Rodents, rats, mice': 25,     # 啮齿动物、大鼠、小鼠
     'Accordion': 26,   # 手风琴
     'Mandolin': 27}    # 洋琵琶


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


def get_txtList():
    # print(txt_name)
    return txt_name


def get_category_to_key():
    # print(category_id)
    return category_id


if __name__ == "__main__":
    args = get_parser()
    mydata = readDataTxt(args.Data_path, "train")

    print(get_txtList())
    print(get_category_to_key())

    for i in range(8):
        print(mydata[i])
    a = list(range(0,30))
    # for name, id in category_id.items():
    #     print(name,id)
    #     a[id] = name
    # print(a)
"""
['c---zaDCTaE', 0, 0, 10]
['fCZi6I6kPpU', 0, 6, 10]
['EV1bVf8Bldk', 0, 0, 10]
['OXvrQ0XIAeM', 0, 0, 10]
['DxQmMOIMRt0', 0, 0, 2]
['lDo2BlqTNhs', 0, 6, 8]
['rqkPh5iYujg', 0, 7, 10]
['WBdWZGDIQ5E', 0, 0, 10]
"""

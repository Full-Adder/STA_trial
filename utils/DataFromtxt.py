import os

txt_name = {"train": "trainSet.txt", "test": "testSet.txt", "val": "valSet.txt", "all": "Annotations.txt"}
category_id = \
    {'Church bell': 0,
     'Male speech, man speaking': 1,
     'Bark': 2,
     'Fixed-wing aircraft, airplane': 3,
     'Race car, auto racing': 4,
     'Female speech, woman speaking': 5,
     'Helicopter': 6,
     'Violin, fiddle': 7,
     'Flute': 8,
     'Ukulele': 9,
     'Frying (food)': 10,
     'Truck': 11,
     'Shofar': 12,
     'Motorcycle': 13,
     'Acoustic guitar': 14,
     'Train horn': 15,
     'Clock': 16,
     'Banjo': 17,
     'Goat': 18,
     'Baby cry, infant cry': 19,
     'Bus': 20,
     'Chainsaw': 21,
     'Cat': 22,
     'Horse': 23,
     'Toilet flush': 24,
     'Rodents, rats, mice': 25,
     'Accordion': 26,
     'Mandolin': 27}


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
    mydata = readDataTxt("../AVE_Dataset", "train")

    print(get_txtList())
    print(get_category_to_key())

    for i in range(8):
        print(mydata[i])

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

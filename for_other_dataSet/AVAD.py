import os
import cv2
import shutil
from moviepy.editor import VideoFileClip
from AVE_modify import mp4_to_jpg, mp4_to_wav, wav_to_h5
from OtherDataSet import get_filename_list_form_txt

data_dir = r"/media/ubuntu/Data/DataSet/AVAD"
mp4_dir = r"/media/ubuntu/Data/DataSet/AVAD/video"
pic_dir = r"/media/ubuntu/Data/DataSet/AVAD/picture"
wav_dir = r"/media/ubuntu/Data/DataSet/AVAD/audio"
h5_dir = r"/media/ubuntu/Data/DataSet/AVAD/h5"
txt_path = os.path.join(data_dir, 'filename.txt')


def gen_txt():
    mp4_list = os.listdir(mp4_dir)
    for i in range(0, len(mp4_list)):
        mp4_path = os.path.join(mp4_dir, mp4_list[i])
        st, en = 0, int(VideoFileClip(mp4_path).duration)
        mp4_list[i] = '&'.join([mp4_list[i].split('.')[0], str(st), str(en)]) + '\n'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(mp4_list)


def modify_AVAD():
    name_list = get_filename_list_form_txt(txt_path)
    for i, data in enumerate(name_list):
        name, st, en = data
        print(i, "now process", name)
        mp4_path = os.path.join(mp4_dir, name + ".mp4")
        if not os.path.exists(mp4_path):
            mp4_path = os.path.join(mp4_dir, name + ".avi")
        mp4_to_jpg(mp4_path, pic_dir, st, en, 4)
        mp4_to_wav(mp4_path, wav_dir, st, en)
        wav_path = os.path.join(wav_dir, name)
        wav_to_h5(wav_path, h5_dir, 4)


def get_from_dir():
    dir_path = os.path.join(data_dir, "a")
    dir_list = os.listdir(dir_path)
    for dir in dir_list:
        print(dir)
        file_dir = os.path.join(dir_path, dir)
        time = 0
        with open(os.path.join(file_dir, "framerate.txt")) as f:
            speed = int(eval(f.readline().strip()))
        img_list = os.listdir(file_dir)
        sorted(img_list)
        for file in img_list:
            print(file)
            if file[-3:] == 'png' and (int(file[-7:-4]) + speed // 2) % speed == 0:
                pic_path = os.path.join(file_dir, file)
                os.makedirs(os.path.join(pic_dir, dir), exist_ok=True)
                move_to_path = os.path.join(pic_dir, dir, "%04d.jpg" % (int(file[-7:-4]) / speed))
                cv2.imwrite(move_to_path, cv2.imread(pic_path))
                print("write file to", move_to_path)
                time += 1
            elif file[-3:] == 'wav':
                wav_path = os.path.join(file_dir, file)
                wav_to_path = os.path.join(wav_dir, file)
                shutil.copyfile(wav_path, wav_to_path)
                print("write file to", wav_to_path)
                wav_to_h5(wav_path[:-4], h5_dir, 4)

        with open(os.path.join(txt_path), "a", encoding='utf-8') as f:
            f.write("&".join([dir, "0", str(time)]) + '\n')


if __name__ == "__main__":
    # gen_txt()
    # a = get_filename_list_form_txt(txt_path)
    # print(a)
    # modify_AVAD()
    get_from_dir()

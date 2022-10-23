import os
from moviepy.editor import VideoFileClip
from AVE_modify import mp4_to_jpg, wav_to_h5
from OtherDataSet import get_filename_list_form_txt

data_dir = r"/media/ubuntu/Data/DataSet/SumMe_ETMD"
mp4_dir = r"/media/ubuntu/Data/DataSet/SumMe_ETMD/video"
pic_dir = r"/media/ubuntu/Data/DataSet/SumMe_ETMD/picture"
wav_dir = r"/media/ubuntu/Data/DataSet/SumMe_ETMD/audio_mono"
h5_dir = r"/media/ubuntu/Data/DataSet/SumMe_ETMD/h5"
txt_path = os.path.join(data_dir, 'filename.txt')


def gen_txt():
    mp4_list = os.listdir(mp4_dir)
    for i in range(0, len(mp4_list)):
        mp4_path = os.path.join(mp4_dir, mp4_list[i])
        st, en = 0, int(VideoFileClip(mp4_path).duration)
        mp4_list[i] = '&'.join([mp4_list[i].split('.')[0], str(st), str(en)]) + '\n'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(mp4_list)


def modify_DATASET():
    name_list = get_filename_list_form_txt(txt_path)
    for i, data in enumerate(name_list):
        name, st, en = data
        mp4_path = os.path.join(mp4_dir, name + ".mp4")
        if not os.path.exists(mp4_path):
            mp4_path = os.path.join(mp4_dir, name + ".avi")
        print(i, len(name_list), "now process", mp4_path)
        mp4_to_jpg(mp4_path, pic_dir, st, en, 4)
        wav_path = os.path.join(wav_dir, name)
        wav_to_h5(wav_path, h5_dir, 4)


if __name__ == "__main__":
    # gen_txt()
    a = get_filename_list_form_txt(txt_path)
    print(a)
    modify_DATASET()

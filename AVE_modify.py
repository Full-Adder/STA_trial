import os
import moviepy.editor as mp
import cv2
import h5py
from scipy import signal
import soundfile as sf
import resampy
import numpy as np
from utils.DataFromtxt import readDataTxt

Data_path = r"/home/ubuntu/AVE_Dataset"
Video_path = r"/home/ubuntu/AVE_Dataset/Video"
Pic_path = r"/media/ubuntu/Data/Picture"
Audio_path = r"/media/ubuntu/Data/Audio"
H5_path = r"/media/ubuntu/Data/H5"


def save_image(image, addr, num, point):
    if point == 2:
        address = os.path.join(addr, "{:0>2d}".format(num) + '.jpg')
    else:
        address = os.path.join(addr, "{:0>4d}".format(num) + '.jpg')
    cv2.imwrite(address, image)


def mp4_to_jpg(video_path, pic_save_path, start_time=0, end_time=10, point=2):
    _, video_name = os.path.split(video_path)
    folder_name = video_name.split('.')[0]  # mp4文件名，无后缀,也是目标地址的文件夹名
    folder_path = os.path.join(pic_save_path, folder_name)  # 新的目录存放每个视频的图片
    os.makedirs(folder_path, exist_ok=True)  # 创建存放视频的对应目录

    video = cv2.VideoCapture(video_path)  # 读入视频文件
    if not video.isOpened():
        return

    fps = video.get(cv2.CAP_PROP_FPS)  # 获取帧率
    print(fps)  # 帧率可能不是整数 需要取整

    rval, frame = video.read()  # videoCapture.read() 函数，第一个返回值为是否成功获取视频帧，第二个返回值为返回的视频帧
    fps_id = 1
    while rval:  # 循环读取视频帧
        if fps_id % round(fps) == round(fps) // 2 and start_time <= fps_id // round(fps) <= end_time:  # 每隔fps帧进行存储操作 ,可自行指定间隔
            save_image(frame, folder_path, fps_id // round(fps), point)
        # cv2.cvWaitKey(1)  # waitKey()--这个函数是在一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)
        rval, frame = video.read()
        fps_id = fps_id + 1
    video.release()
    print('save_success ' + folder_path)


def generate_jpg(mode):
    data_list = readDataTxt(Data_path, mode)
    for data in data_list:
        mp4_to_jpg(os.path.join(Video_path, data[0] + ".mp4"), Pic_path)  # , data[-2], data[-1])


def mp4_to_wav(videos_path, to_path, start_time=0, end_time=10):
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    my_clip = mp.VideoFileClip(videos_path)
    if end_time - start_time != 10:
        my_clip.subclip(start_time, end_time)
    _, videos_name = os.path.split(videos_path)
    audio_path = os.path.join(to_path, videos_name[:-4]+".wav")
    my_clip.audio.write_audiofile(audio_path)


def generate_wav(mode):
    if not os.path.exists(Audio_path):
        os.makedirs(Audio_path)
    data_list = readDataTxt(Data_path, mode)
    for data in data_list:
        mp4_to_wav(os.path.join(Video_path, data[0]+ ".mp4"), Audio_path)  # data[-2], data[-1]


def wav_to_h5(audio_path, to_path, point=2):
    Audio_file_path = audio_path + ".wav"
    _, Audio_name = os.path.split(audio_path)
    os.makedirs(os.path.join(to_path, Audio_name), exist_ok=True)
    samples, samplerate = sf.read(Audio_file_path)
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)
    SAMPLE_RATE = 16000
    if samplerate != SAMPLE_RATE:
        samples = resampy.resample(samples, samplerate, SAMPLE_RATE)  # 采样速率转换44100->16000

    num = samples.size // SAMPLE_RATE
    for picp in range(0, num):
        s = picp * 16000
        e = (picp + 1) * 16000
        resamples = samples[s:e]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        _, _, spectrogram = signal.spectrogram(resamples, SAMPLE_RATE, nperseg=160, noverlap=80)
        spectrogram = np.log(spectrogram + 1e-7)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        audio_output = np.divide(spectrogram - mean, std + 1e-9)  # 257，61
        if point == 2:
            h5_path = os.path.join(to_path, Audio_name, "{:0>2d}".format(picp) + ".h5")
        else:
            h5_path = os.path.join(to_path, Audio_name, "{:0>4d}".format(picp) + ".h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("dataset", data=audio_output)
    print(Audio_name, ".h5 is ok")


def generate_h5(mode='train'):
    if not os.path.exists(H5_path):
        os.makedirs(H5_path)
    data_list = readDataTxt(Data_path, mode)
    for data in data_list:
        wav_to_h5(os.path.join(Audio_path, data[0]), H5_path)


def modefy_data():
    generate_jpg("train")
    generate_jpg("test")
    generate_wav("train")
    generate_wav("test")
    generate_h5("train")
    generate_h5("test")


if __name__ == "__main__":
    modefy_data()

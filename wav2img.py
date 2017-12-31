import wave
import matplotlib.pyplot as plt
import numpy as np
import os

# audio_path = "./data/audio/"
# class_name = os.listdir(audio_path)
# class_num = len(class_name)
# for i in range(class_num):
#     audio_name = os.listdir(audio_path+class_name[i])
#     audio_num = len(audio_name)
#     for j in range(audio_num):
#         file = wave.open(audio_path+class_name[i]+'/'+audio_name[j], 'rb')
#         frame_rate = file.getframerate()  # get the frame rate of the .wav file
#         frame_num = file.getnframes()  # get the number of frames in the .wav file
#         strData = file.readframes(frame_num)
#         waveData = np.fromstring(strData, dtype=np.int16)
#         waveData = waveData * 1.0 / (max(abs(waveData)))  # normalize the data
#         waveData = np.reshape(waveData, [1, frame_num])
#         # print(waveData.shape)
#         file.close()

#         # plot the wave
#         plt.axes([0, 0, 1, 1])
#         plt.specgram(waveData[0], Fs=frame_rate, scale_by_freq=True, sides='default', cmap='Greys')  # draw the spectrogram
#         plt.axis('off')
#         fig = plt.gcf()
#         plt.show()
#         fig.set_size_inches(1, 1)  # set the img size to 1 inch by 1 inch
#         img_name = class_name[i]+'_'+str(j)+'.jpg'
#         fig.savefig('./data/spectrogram/'+img_name, dpi=128, frameon=False)



audio_path = "./data/clips/"
audio_names = os.listdir(audio_path)
audio_num = len(audio_names)
for j in range(audio_num):
    file = wave.open(audio_path+audio_names[j], 'rb')
    frame_rate = file.getframerate()  # get the frame rate of the .wav file
    frame_num = file.getnframes()  # get the number of frames in the .wav file
    strData = file.readframes(frame_num)
    waveData = np.fromstring(strData, dtype=np.int16)
    waveData = waveData * 1.0 / (max(abs(waveData)))  # normalize the data
    waveData = np.reshape(waveData, [1, frame_num])
    # print(waveData.shape)
    file.close()

    # plot the wave
    plt.axes([0, 0, 1, 1])
    plt.specgram(waveData[0], Fs=frame_rate, scale_by_freq=True, sides='default', cmap='Greys')  # draw the spectrogram
    plt.axis('off')
    fig = plt.gcf()
    # plt.show()
    fig.set_size_inches(1, 1)  # set the img size to 1 inch by 1 inch
    # img_name = class_name[i]+'_'+str(j)+'.jpg'
    img_name = audio_names[j].replace('.wav', '.jpg')
    fig.savefig('./data/spectrogram/'+img_name, dpi=128, frameon=False)
    plt.close('all')
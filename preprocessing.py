import os
import  wave


audio_path = './data/source_songs/'
class_name = os.listdir(audio_path)
class_num = len(class_name)
for i in range(class_num):
    count = 0
    audio_names = os.listdir(audio_path+class_name[i])
    audio_num = len(audio_names)
    for j in range(audio_num):
        file_path = audio_path+class_name[i]+'/'+audio_names[j]
        file = wave.open(file_path)
        frame_rate = file.getframerate()
        frame_num = file.getnframes()
        duration = int(frame_num/frame_rate)
        slice_num = duration//5
        start_pos = 0
        for _ in range(slice_num):
            out_path = './data/audio/'+class_name[i]+'/'+class_name[i]+'_'+str(count)+'.wav'
            count += 1
            os.system('./preprocessing.sh '+file_path+' '+out_path+' '+str(start_pos))
            start_pos += 5


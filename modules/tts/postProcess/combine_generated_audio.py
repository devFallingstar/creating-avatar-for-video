import matplotlib.pyplot as plt
import librosa.display
from datetime import datetime
import os
import re
import librosa
import math
import numpy as np
import scipy.io.wavfile
import soundfile as sf


subtitle_address="/home/nebula/PycharmProjects/deepvoice3_pytorch/subtitle.txt"
audio_folder_path="/home/nebula/PycharmProjects/deepvoice3_pytorch/sample"


# Loading filename of the audio and sorting
arr = os.listdir(audio_folder_path)
arr.sort(key=lambda f: int(re.sub('\D', '', f)))

with open(subtitle_address) as f:
    # reading subtitle and removing unwanted spaces
    content = f.readlines()
    content = [x.strip() for x in content]


    x=1
    next=False
    # finding the last subtitle and reading the endtime
    content.reverse()
    endtime = content[1]
    endtime = endtime[17:]
    # converting time into seconds
    pt = datetime.strptime(endtime, '%H:%M:%S,%f')
    total_seconds = pt.second + pt.minute * 60 + pt.hour * 3600+1
    content.reverse()

    # making a zero array of the length of the audio
    endtime = np.zeros(math.floor(total_seconds*22050))

    #reading each line of the subtitle and looking for number eg: 1 ,2 ,3
    # and using the variable 'x' to look for the correct number
    #  after finding the correct number , assigning variable 'next' as True
    #  Because in subtitle after number comes the time , we are looking for the time
    # example of subtitle is given below
    # ##############################################################################################
    # 1
    # 00: 00:05, 480 --> 00: 00:07, 060
    # What is that? Fumigation?
    #
    # 2
    # 00: 00:0
    # 8, 860 --> 00: 00:0
    # 9, 750
    # Close the windows

    #########################################################################################
    for line in content:
        if next == True:
            start=line[:-17]
            end=line[17:]

            # changing starting time to seconds
            pt = datetime.strptime(str(start), '%H:%M:%S,%f')
            start = pt.second + pt.minute * 60 + pt.hour * 3600
            pt = datetime.strptime(str(end), '%H:%M:%S,%f')
            end = pt.second + pt.minute * 60 + pt.hour * 3600

            # Loading the audio using librosa
            y, sr = librosa.load(audio_folder_path+"/"+arr[x-2])

            # adding silence if the audio to small
            if ((end - start) > len(y) / 22050):
                fd = 0.025
                f_size = math.floor(fd * sr)
                n_f = math.floor(y.shape[0] / f_size)
                alpha = 1
                data = []
                newarray = np.array_split(y, n_f)
                for i in newarray:
                    # print(np.amax(i))
                    if (np.amax(i) > 0.025):
                        if alpha == 1:
                            b = np.copy(i)
                            # print(b.shape)
                            alpha = alpha + 1
                            continue
                        b = np.concatenate([b, i])
                    else:
                        b = np.concatenate([b, i])
                        b = np.concatenate([b, i])

                y=b



            # copying the audio to  the zero array at the specific time
            for i in range(len(y)) :
                endtime[(start*22050)+i] = y[i]

            next = False



        if line == str(x):
            # print(line)
            next=True
            x=x+1

    # To visualize the audio
    plt.subplot(1, 1, 1)
    librosa.display.waveplot(endtime, sr=sr, color='b', alpha=0.8)
    plt.title('Original')
    plt.tight_layout()
    plt.show()

    # to write the combined audio
    # librosa.output.write_wav('combined_audio.wav', endtime, sr)
    sf.write("combined_audio.wav",endtime,sr,format='WAV',endian='LITTLE',subtype='PCM_16')
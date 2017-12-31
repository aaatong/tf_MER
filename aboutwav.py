'''

The first code section is a backup of the basic version

'''

import wave
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# filepath = "./data/source_songs/1/"
# filename = os.listdir(filepath)
# f = wave.open(filepath + filename[0], 'rb')
f = wave.open('0_5.wav', 'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
# print(nframes/framerate)
strData = f.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.int16)
waveData = waveData * 1.0 / (max(abs(waveData)))
print(waveData.shape, nframes, framerate)
waveData = np.reshape(waveData, [nframes, 1]).T
print(waveData.shape)
f.close()
# plot the wave
plt.axes([0, 0, 1, 1])
spec, freqs, t, im = plt.specgram(waveData[0], Fs=framerate, scale_by_freq=True, sides='default', cmap='Greys', NFFT=512)
print(spec.shape, freqs.shape, t.shape)
plt.axis('off')
fig = plt.gcf()
plt.show()
fig.set_size_inches(1, 1)
fig.savefig('test.jpg', dpi=128, frameon=False)
image = mpimg.imread('test.jpg')
image = np.array(image)
print(image.shape)
test = image[:,:,0]
np.save('test.npy', test)
# plt.imsave(image[:][:][0], 'gray.jpg') 16384 
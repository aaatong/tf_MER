import os
import numpy as np
import matplotlib.image as mpimg


if os.path.exists('data/npz/dataset.npz'):
    os.system('rm data/npz/dataset.npz')
img_path = './data/spectrogram/'
img_names = os.listdir(img_path)
img_num = len(img_names)
features = np.zeros((img_num, 128, 128))
labels = np.zeros(img_num)
for i in range(img_num):
    image = mpimg.imread(img_path+img_names[i])
    image = np.array(image)
    # print(image.shape)
    temp = image[:, :, 0]
    # print(temp.shape)
    features[i] = temp
    labels[i] = int(img_names[i][0])
print(features.shape)
np.savez('./data/npz/dataset.npz', features=features, labels=labels)

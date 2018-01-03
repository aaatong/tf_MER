import os
import numpy as np
import matplotlib.image as mpimg


num_classes = 2

if os.path.exists('data/npz/dataset.npz'):
    os.system('rm data/npz/dataset.npz')
img_path = './data/spectrogram/'
img_names = os.listdir(img_path)
img_num = len(img_names)
features = np.zeros((img_num, 128, 256))    # img_num*h*w
labels = np.zeros((img_num, num_classes))
for i in range(img_num):
    image = mpimg.imread(img_path+img_names[i])
    image = np.array(image)
    # print(image.shape)
    temp = image[:, :, 0]
    # print(temp.shape)
    features[i] = temp
    one_hot = int(img_names[i][0])
    labels[i][one_hot] = 1
print(features.shape)
np.savez('./data/npz/dataset.npz', features=features, labels=labels)

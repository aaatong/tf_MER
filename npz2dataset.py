import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

num_classes = 8

with np.load('./data/npz/8_classes.npz') as data:
    features = data['features']
    labels = data['labels']
# labels = labels - 1   # labels should be named from 0
features, labels = shuffle(features, labels, random_state=0)
test_num = features.shape[0]//10
test_x = features[0:test_num]
test_y = labels[0:test_num]
dataset = tf.data.Dataset.from_tensor_slices((features[test_num:], labels[test_num:]))
dataset = dataset.repeat()
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()
# print(test_y)
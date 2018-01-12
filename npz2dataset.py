import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

num_classes = 4
batch_size = 64

with np.load('./data/npz/test.npz') as data:
    features = data['features']
    labels = data['labels']
# labels = labels - 1   # labels should be named from 0
features, labels = shuffle(features, labels, random_state=0)
test_num = features.shape[0]//10
# test_x = features[0:test_num]
# test_y = labels[0:test_num]
# dataset = tf.data.Dataset.from_tensor_slices((features[test_num:], 
#                                               labels[test_num:]))
# dataset = dataset.repeat()
# dataset = dataset.batch(batch_size)
# iterator = dataset.make_initializable_iterator()
# next_batch = iterator.get_next()
# print(test_y)

test_dataset = tf.data.Dataset.from_tensor_slices((features[0:test_num], 
                                                   labels[0:test_num]))
test_dataset = test_dataset.batch(batch_size)
test_batch_num = test_num//batch_size + 1

training_dataset = tf.data.Dataset.from_tensor_slices((features[test_num:], 
                                                       labels[test_num:]))
training_dataset = training_dataset.repeat()
training_dataset = training_dataset.batch(batch_size)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types, 
                                           training_dataset.output_shapes)
next_batch = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)
print(features.shape)
print(labels.shape)
print(test_num)
print(test_batch_num)
print(test_dataset)
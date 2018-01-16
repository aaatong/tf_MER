import tensorflow as tf

# pred = tf.placeholder(tf.float32, [5])
# label = [0, 1, 1, 0, 0]
# stream = [[0,0,0,0,0],[0,1,1,0,0],[1,0,0,0,1]]
# acc, op = tf.metrics.accuracy(pred, label)

# with tf.Session() as sess:
#     sess.run(tf.local_variables_initializer())
#     for i in range(3):
#         rst = sess.run((acc, op), feed_dict={pred:stream[i]})
#     print(rst[1])

# sess = tf.Session()
# max_value = tf.placeholder(tf.int64, shape=[])
# dataset = tf.data.Dataset.range(max_value)
# dataset = dataset.batch(6)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()

# # Initialize an iterator over a dataset with 10 elements.
# sess.run(iterator.initializer, feed_dict={max_value: 10})
# for i in range(10):
#   value = sess.run(next_element)
#   print(value)

# import numpy as np
# test_num = 100
# batch_size = 64
# features = np.ones([200])
# labels = np.ones([200])
# test_dataset = tf.data.Dataset.from_tensor_slices((features[0:test_num], 
#                                                    labels[0:test_num]))
# test_dataset = test_dataset.batch(batch_size)
# test_batch_num = test_num//batch_size

# training_dataset = tf.data.Dataset.from_tensor_slices((features[test_num:], 
#                                                        labels[test_num:]))
# training_dataset = training_dataset.repeat()
# training_dataset = training_dataset.batch(batch_size)

# iterator = tf.data.Iterator.from_structure(training_dataset.output_types, 
#                                            training_dataset.output_shapes)
# next_batch = iterator.get_next()
# training_init_op = iterator.make_initializer(training_dataset)
# test_init_op = iterator.make_initializer(test_dataset)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(training_init_op)
#     for i in range(5):
#         train_x, train_y = next_batch
#         print(sess.run((tf.shape(train_x), tf.shape(train_y))))

#     sess.run(test_init_op)
#     test_x, test_y = next_batch
#     print(test_batch_num)
#     for i in range(test_batch_num+1):
#         print('---------------------------------------------------------------')
        
#         print(sess.run((tf.shape(test_x), tf.shape(test_y))))




# # Create some variables.
# v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

# inc_v1 = v1.assign(v1+1)
# dec_v2 = v2.assign(v2-1)

# # Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()

# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# # Later, launch the model, initialize the variables, do some work, and save the
# # variables to disk.
# with tf.Session() as sess:
#   sess.run(init_op)
#   # Do some work with the model.
#   for i in range(10):
#     inc_v1.op.run()
#     dec_v2.op.run()
#     # Save the variables to disk.
#     if i==5 or i==7:
#         save_path = saver.save(sess, "../tmp/my-model", global_step=i)
#         print("Model saved in file: %s" % save_path)


# tf.reset_default_graph()

# # Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])

# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "../tmp/my-model-7")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())

# import npz2dataset as nd
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file('../tmp/my-model', tensor_name='fc1/b_fc1', all_tensors=False)


# tf.reset_default_graph()
# imported_meta = tf.train.import_meta_graph("../tmp/my-model.meta")  

# batch_x, batch_y = nd.next_batch
# with tf.Session() as sess:
#     # Restore variables from disk
#     #   sess.run(tf.global_variables_initializer())
#     imported_meta.restore(sess, "../tmp/my-model")
#     #   saver.restore(sess, "../tmp/my-model")
#     #   imported_meta.restore(sess, tf.train.latest_checkpoint('../tmp/'))
#     print("Model restored.")
#     # Check the values of the variables
#     train_x, train_y = sess.run((batch_x, batch_y))
#     x_input = tf.get_default_graph().get_tensor_by_name('x_input:0')
#     y_input = tf.get_default_graph().get_tensor_by_name('y_input:0')  
#     train_step = tf.get_default_graph().get_operation_by_name('optimizer/train_step')
#     #   print(sess.run(x_input))
#     sess.run(train_step, feed_dict={x_input: train_x, y_input: train_y})

import time
import numpy as np
from sklearn.utils import shuffle

with np.load('./data/npz/dataset.npz') as data:
    features = data['features'].astype(np.float32)
    labels = data['labels'].astype(np.float32)

# print(data.dtype)
print(labels.dtype)
time.sleep(20)
print('tick')
# features, labels = shuffle(features, labels, random_state=0)
# test_num = features.shape[0]//10

# features_ph = tf.placeholder(tf.float32, shape=(features.shape[0], 128, 256))
# features_v = tf.Variable(features_ph)
# dataset = tf.data.Dataset.from_tensor_slices((features_v, labels))

# iterator = tf.data.Iterator.from_structure(dataset.output_types, 
#                                            dataset.output_shapes)
# next_batch = iterator.get_next(name='next_batch')
# training_init_op = iterator.make_initializer(dataset)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer(), feed_dict={features_ph: features})
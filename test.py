import tensorflow as tf

pred = tf.placeholder(tf.float32, [5])
label = [0, 1, 1, 0, 0]
stream = [[0,0,0,0,0],[0,1,1,0,0],[1,0,0,0,1]]
acc, op = tf.metrics.accuracy(pred, label)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    for i in range(3):
        rst = sess.run((acc, op), feed_dict={pred:stream[i]})
    print(rst[1])

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

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


# # tf.reset_default_graph()

# # Create some variables.
# # b_conv1 = tf.Variable(tf.constant(0.0, shape=[16]), name='b_conv1')
# cross_entropy = tf.get_variable('cross_entropy', shape=[])

# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "../tmp/my-model")
#   print("Model restored.")
#   # Check the values of the variables
#   print(sess.run(b_conv1))

i = 2
if i in [0,1,2,3,4]:
    print('tick')
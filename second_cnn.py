import time
import sys
import tensorflow as tf
import npz2dataset as nd

def cnn(x):
    """ A simple cnn built with tensorflow 
        3 convolution layers
        no dropout layer
    """
    with tf.name_scope('reshape'):
        x_input = tf.reshape(x, [-1, 128, 256, 1], name='x_input')

    with tf.name_scope('conv1'):
        w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1), name='w_conv1')
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name='b_conv1')
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')+b_conv1, name='relu')

    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.name_scope('conv2'):
        w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name='w_conv2')
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv2')
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')+b_conv2, name='relu')

    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.name_scope('conv3'):
        w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='w_conv3')
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv3')
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')+b_conv3, name='relu')

    with tf.name_scope('pool2'):
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.name_scope('fc1'):
        w_fc1 = tf.Variable(tf.truncated_normal([16*32*64, 1024], stddev=0.1), name='w_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16*32*64], name='flatten')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1, name='relu')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob, name='dropout')

    with tf.name_scope('fc2'):
        w_fc2 = tf.Variable(tf.truncated_normal([1024, nd.num_classes], stddev=0.1), name='w_fc2')
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[nd.num_classes]), name='b_fc2')
        # cnn_output = tf.matmul(h_fc1_drop, w_fc2)+b_fc2
        cnn_output = tf.matmul(h_fc1, w_fc2)+b_fc2

    return cnn_output, keep_prob

def main():
    num_steps = 2
    x_input = tf.placeholder(tf.float32, [None, 128, 256], name='x_input')
    y_input = tf.placeholder(tf.int32, [None, nd.num_classes], name='y_input')

    # depth = tf.placeholder(tf.int32)
    # labels = tf.one_hot(y_input, nd.num_classes)

    cnn_output, keep_prob = cnn(x_input)

    with tf.name_scope('loss'):
        # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=cnn_output)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=cnn_output)
    
    cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy, name='train_step')
        # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(cnn_output, 1), tf.argmax(y_input, 1))
        correct_pred = tf.cast(correct_pred, tf.float32)
    accuracy = tf.reduce_mean(correct_pred, name='accuracy')
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('tb_log/train', sess.graph)
        # test_writer = tf.summary.FileWriter('tb_log/test')
        sess.run(tf.global_variables_initializer(), feed_dict={nd.features_ph: nd.features[nd.test_num:]})
        # sess.run(tf.local_variables_initializer())
        # sess.run(nd.iterator.initializer)
        sess.run(nd.training_init_op)
        # sess.graph.finalize()
        # batch_x, batch_y = nd.next_batch
        # batch_x = batch_x.eval()
        # batch_y = batch_y.eval()
        batch_x, batch_y = nd.next_batch
        for i in range(1, num_steps+1):
            train_x, train_y = sess.run((batch_x, batch_y))
            # batch_x, batch_y = nd.next_batch
            # batch_x = batch_x.eval()    # convert a tensor to a numpy array
            # batch_y = batch_y.eval()
            if i % 100 == 0 or i == 1:
                summary, train_accuracy, loss = sess.run([merged, accuracy, cross_entropy], feed_dict={x_input: train_x, y_input: train_y, keep_prob: 0.5})
                train_writer.add_summary(summary, i)
                print('step %d, training accuracy %g, loss %g' % (i, train_accuracy, loss))
                sys.stdout.flush()
            sess.run(train_step, feed_dict={x_input: train_x, y_input: train_y, keep_prob: 0.5})


        save_path = saver.save(sess, "../tmp/my-model")
        print("Model saved in file: %s" % save_path)

        sess.run(nd.test_init_op)
        test_accuracy, acc_update_op = tf.metrics.accuracy(labels=tf.argmax(y_input, 1), predictions=tf.argmax(cnn_output, 1), name='metrics')
        # metrics = {
        #     'accuracy': test_accuracy,
        #     'acc_op': acc_update_op
        # }
        batch_x, batch_y = nd.next_batch
        for i in range(nd.test_batch_num):
            print('tick')
            sess.run(tf.local_variables_initializer())
            
            # test_x, test_y = (batch_x, batch_y).eval()
            test_x, test_y = sess.run((batch_x, batch_y))
            # test_y = batch_y.eval()
            
            # print(sess.run(tf.shape(cnn_output),feed_dict={x_input: test_x, y_input: test_y, keep_prob: 1.0, phase_train: False}))
            # print(sess.run(tf.shape(test_y)))
            rst = sess.run((test_accuracy, acc_update_op), feed_dict={
                x_input: test_x, y_input: test_y, keep_prob: 1.0})
        print('----------------------------------test accuracy %g' % rst[1])
        # test_accuracy = sess.run(accuracy, feed_dict={x_input: nd.test_x, y_input: nd.test_y, keep_prob: 1.0})
        # print('test accuracy %g' % test_accuracy)
        # print('test accuracy %g' % accuracy.eval(feed_dict={x_input: nd.test_x, y_input: nd.test_y, keep_prob: 1.0}))

if __name__ == '__main__':
    log = open('log.txt', 'a')
    sys.stdout = log
    # tf.app.run(main=main)
    print('-------------------------start-------------------------')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    main()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print('--------------------------end--------------------------')
    log.close()
    

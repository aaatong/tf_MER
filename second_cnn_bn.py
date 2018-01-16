import time
import sys
import tensorflow as tf
import npz2dataset as nd

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('batch_normalization'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                       name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                        name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # pred = tf.constant(True, dtype=tf.bool)
        mean, var = tf.cond(phase_train, mean_var_with_update, 
                            lambda: (ema.average(batch_mean), 
                                     ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def cnn(x, keep_prob, phase_train):
    """ A simple cnn built with tensorflow 
        3 convolution layers
        no dropout layer
    """
    with tf.name_scope('reshape'):
        x_input = tf.reshape(x, [-1, 128, 256, 1], name='x_input')

    with tf.name_scope('conv_layer_1'):
        w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.3), name='w_conv1')
        b_conv1 = tf.Variable(tf.constant(0.0, shape=[16]), name='b_conv1')
        conv1 = tf.nn.conv2d(x_input, w_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')+b_conv1
        conv1_bn = batch_norm(conv1, 16, phase_train)
        h_conv1 = tf.nn.relu(conv1_bn, name='relu')

    with tf.name_scope('pool_layer_1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.name_scope('conv_layer_2'):
        w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.3), name='w_conv2')
        b_conv2 = tf.Variable(tf.constant(0.0, shape=[32]), name='b_conv2')
        conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')+b_conv2
        conv2_bn = batch_norm(conv2, 32, phase_train)
        h_conv2 = tf.nn.relu(conv2_bn, name='relu')

    with tf.name_scope('pool_layer_2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.name_scope('conv_layer_3'):
        w_conv3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.3), name='w_conv3')
        b_conv3 = tf.Variable(tf.constant(0.0, shape=[64]), name='b_conv3')
        conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')+b_conv3
        conv3_bn = batch_norm(conv3, 64, phase_train)
        h_conv3 = tf.nn.relu(conv3_bn, name='relu')

    with tf.name_scope('pool_layer_3'):
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.name_scope('fc1'):
        w_fc1 = tf.Variable(tf.truncated_normal([16*32*64, 1024], stddev=0.1), name='w_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
        h_pool3_flat = tf.reshape(h_pool3, [-1, 16*32*64], name='flatten')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1, name='relu')

    with tf.name_scope('dropout_layer'):
        # keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob, name='dropout')

    with tf.name_scope('fc2'):
        w_fc2 = tf.Variable(tf.truncated_normal([1024, nd.num_classes], stddev=0.1), name='w_fc2')
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[nd.num_classes]), name='b_fc2')
        cnn_output = tf.matmul(h_fc1_drop, w_fc2)+b_fc2
        # cnn_output = tf.matmul(h_fc1, w_fc2)+b_fc2

    return cnn_output, keep_prob

def main():
    num_steps = 2
    x_input = tf.placeholder(tf.float32, [None, 128, 256])
    y_input = tf.placeholder(tf.int32, [None, nd.num_classes])

    # depth = tf.placeholder(tf.int32)
    # labels = tf.one_hot(y_input, nd.num_classes)
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)
    cnn_output, keep_prob = cnn(x_input, keep_prob, phase_train)

    with tf.name_scope('loss'):
        # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=cnn_output)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=cnn_output)
    
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(cnn_output, 1), tf.argmax(y_input, 1))
        correct_pred = tf.cast(correct_pred, tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        def run_test():
            sess.run(nd.test_init_op)
            test_accuracy, acc_update_op = tf.metrics.accuracy(
                labels=tf.argmax(y_input, 1), 
                predictions=tf.argmax(cnn_output, 1))
            # metrics = {
            #     'accuracy': test_accuracy,
            #     'acc_op': acc_update_op
            # }
            batch_x, batch_y = nd.next_batch
            for i in range(nd.test_batch_num):
                # print('tick')
                sess.run(tf.local_variables_initializer())
                
                # test_x, test_y = (batch_x, batch_y).eval()
                test_x, test_y = sess.run((batch_x, batch_y))
                # test_y = batch_y.eval()
                
                # print(sess.run(tf.shape(cnn_output),feed_dict={x_input: test_x, y_input: test_y, keep_prob: 1.0, phase_train: False}))
                # print(sess.run(tf.shape(test_y)))
                rst = sess.run((test_accuracy, acc_update_op), feed_dict={
                    x_input: test_x, y_input: test_y, keep_prob: 1.0, 
                    phase_train: False})
            print('----------------------------------test accuracy %g' % rst[1])

            sess.run(nd.training_init_op)
            


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
            # batch_x = batch_x.eval()
            # batch_y = batch_y.eval()
            if i % 100 == 0 or i == 1:
                # print(sess.run(cnn_output, feed_dict={x_input: batch_x, keep_prob: 1.0}))
                # print(batch_y)
                # return
                summary, train_accuracy, loss = sess.run(
                    [merged, accuracy, cross_entropy], 
                    feed_dict={x_input: train_x, y_input: train_y, 
                               keep_prob: 1.0, phase_train: False})
                train_writer.add_summary(summary, i)
                print('step %d, training accuracy %g, loss %g' 
                    % (i, train_accuracy, loss))
                sys.stdout.flush()
            sess.run(train_step, feed_dict={x_input: train_x, y_input: train_y, 
                                            keep_prob: 0.5, 
                                            phase_train: True})
            if i % 1000 == 0:
                run_test()



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
    

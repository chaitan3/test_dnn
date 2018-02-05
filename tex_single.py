#!/usr/bin/python2

from gen_latex import get_data

train_data, test_data = get_data()

import tensorflow as tf

sess = tf.Session()

n_symbols = 71
n_train = len(train_data[1])
n_test = len(test_data[1])
x = tf.placeholder(tf.float32, shape=[None, 28, 28])
y_ = tf.placeholder(tf.float32, shape=[None, n_symbols])
keep_prob = tf.placeholder(tf.float32)

def multi_layer():
    def weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    xi = tf.reshape(x, [-1, 28, 28, 1])
    W = weights([3, 3, 1, 64])
    b = bias([64])
    h = tf.nn.relu(tf.nn.conv2d(xi, W, strides=[1,1,1,1], padding='SAME') + b)
    W = weights([3, 3, 64, 32])
    b = bias([32])
    h = tf.nn.relu(tf.nn.conv2d(h, W, strides=[1,1,1,1], padding='SAME') + b)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W = weights([5, 5, 32, 64])
    b = bias([64])
    h = tf.nn.relu(tf.nn.conv2d(h, W, strides=[1,1,1,1], padding='SAME') + b)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W = weights([7*7*64, 1024])
    b = bias([1024])
    h = tf.reshape(h, [-1, 7*7*64])
    h = tf.nn.relu(tf.matmul(h, W) + b)

    h = tf.nn.dropout(h, keep_prob)
    W = weights([1024, n_symbols])
    b = bias([n_symbols])
    y = tf.matmul(h, W) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        #print i
        start = i*100 % (n_train - n_train % 100)
        end = start + 100
        batch = train_data[0][start:end], train_data[1][start:end]
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            #import pdb;pdb.set_trace()
            batch = test_data
            print('test', sess.run(acc, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
            batch = train_data
            print('train', sess.run(acc, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))

if __name__ == "__main__":
    multi_layer()

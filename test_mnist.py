import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def mnist_single_layer():
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    sess.run(tf.global_variables_initializer())

    y = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    for i in range(1000):
        print i
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))
    print sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

def mnist_multi_layer():
    def weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    def bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    xi = tf.reshape(x, [-1, 28, 28, 1])
    W = weights([3, 3, 1, 32])
    b = bias([32])
    h = tf.nn.relu(tf.nn.conv2d(xi, W, strides=[1,1,1,1], padding='SAME') + b)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W = weights([5, 5, 32, 64])
    b = bias([64])
    h = tf.nn.relu(tf.nn.conv2d(h, W, strides=[1,1,1,1], padding='SAME') + b)
    h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W = weights([7*7*64, 1024])
    b = bias([1024])
    h = tf.reshape(h, [-1, 7*7*64])
    h = tf.nn.relu(tf.matmul(h, W) + b)

    W = weights([1024, 10])
    b = bias([10])
    y = tf.matmul(h, W) + b


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        print i
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
        if i % 100 == 0:
            print sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

if __name__ == "__main__":
    mnist_multi_layer()

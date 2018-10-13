from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os

import argparse
import sys
data_dir = ''   # 样本数据存储的路径
log_dir = ''    # 输出日志保存的路径


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# input data
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)  # runing on server

learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 50

n_input = 784
n_classes = 10
dropout = 0.75
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, n_input],name = 'x_input_data')
    y = tf.placeholder(tf.float32, [None, n_classes], name = 'y_label_data')
keep_prob = tf.placeholder(tf.float32) # dropout


def conv2d(name, x, W, b, s=1):
    with tf.name_scope(name):
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME'))

def maxpool2d(name, x, k=2, s=2):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding='VALID', name=name)

def norm(name, l_input, lsize=4):
    with tf.name_scope(name):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,
                     beta=0.75, name=name)


def conv_net(x, weights, biases, dropout):
    # Reshape input picture

    with tf.name_scope('conv_net'):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'], s=1)
        tf.summary.histogram('wc1', weights['wc1'])
        tf.summary.histogram('bc1',biases['bc1'])
        tf.summary.histogram('covnv1', conv1)
        print('cov1.shape: ', conv1.get_shape().as_list())
        pool1 = maxpool2d('pool1', conv1, k=2, s=2)
        print('pool1.shape: ', pool1.get_shape().as_list())
        norm1 = norm('norm1', pool1)

        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'], s=1)
        tf.summary.histogram('wc2', weights['wc2'])
        tf.summary.histogram('bc2',biases['bc2'])
        tf.summary.histogram('covnv2', conv2)
        pool2 = maxpool2d('pool2', conv2, k=2, s=2)
        print('pool2.shape: ', pool2.get_shape().as_list())
        norm2 = norm('pool2', pool2)

        fc1 = tf.reshape(norm2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        tf.summary.histogram('fc1',fc1)

        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)
        tf.summary.histogram('fc2', fc2)

        out = tf.matmul(fc2, weights['out']) + biases['out']
        tf.summary.histogram('out', out)
        return out

with tf.name_scope('Weights'):
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),

        'wd1': tf.Variable(tf.random_normal([7 * 7 * 128, 1024])),
        'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

with tf.name_scope('Biases'):
    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),

        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

with tf.name_scope('pred'):
    pred = conv_net(x, weights, biases, keep_prob)
    #tf.summary.scalar('pred', pred)
with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    tf.summary.scalar('cost', cost)
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope('Correct_pred'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) #  return  A `Tensor` of type `bool`
    # tf.summary.scalar('correct_pred', correct_pred)
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()

# for v in tf.trainable_variables():
#     tf.summary.histogram(v.name, v)

with tf.Session(config=config) as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("logs", sess.graph)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #        # 执行反向传播优化过程
        sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y,
                                 keep_prob: dropout})
        # train_writer.add_summary(rs, step)

        if step % display_step == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                    y: batch_y,
                                                    keep_prob: 1.}) #metadata记录内存，cpu等
            # tf.summary.scalar('loss',loss)
            # tf.summary.scalar('acc', acc)
            rs = sess.run(merged, feed_dict={x: batch_x, y: batch_y},
                                 options=run_options, run_metadata=run_metadata)
            #train_writer.add_run_metadata(run_metadata, step)
            train_writer.add_summary(rs, step)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
        step += 1



    print("Optimization Finished!")

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                   y: mnist.test.labels[:256],
                                   keep_prob: 1.}))

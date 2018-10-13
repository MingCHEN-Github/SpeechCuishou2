from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()

data_dir = ''   # 样本数据存储的路径
log_dir = ''    # 输出日志保存的路径

#config.gpu_options.allow_growth = True
#超参数的定义
n_inputs = 26               # MFCC数据输入为40，fBank 为26
n_classes = 16               #输入语音的类别数量
dropout = 0.5
learning_rate = 0.001 #用Adam优化的时候，通常取lr=0.001可达到较好效果
training_iters = 2000
batch_size = 32
display_step = 50
#载入已提取的特征集，和 标签集
# 320*t*26维（语音条数，时长，单帧特征数）
train_features = np.load('C:\\Users\\zkycs3\\Desktop\\SpeechCuishou1\\speechCNN\\train_8k_fbank_features.npy')
train_labels = np.load('C:\\Users\\zkycs3\\Desktop\\SpeechCuishou1\\speechCNN\\train_8k_labels.npy')
#把train_lables转换为 独热编码
#train_labels = tf.one_hot(indices=train_labels,depth=n_classes)

# 计算时间最长的那条语音的帧数，统一每条语音的特征维数，使其一样长
wav_max_len = max([len(feature) for feature in train_features])
print("wav_max_len=",wav_max_len)

#把标签[1,2,3],转换为[[1,0,0],[0,1,0],[0,0,1]这样的形式
def lables_onehot(lables):
    one_hot_index = np.arange(len(lables))*len(lables)+lables
    one_hot = np.zeros(len(lables),len(lables))
    one_hot.flat[one_hot_index]=1
    return one_hot
#特征补0预处理，使得每个输入的特征向量第2个维数相同，第3个维数都为26
def fill_zero(features,n_inputs=26):
    # 填充0
    features_data = []
    for mfccs in features:
        while len(mfccs) < wav_max_len:
            mfccs.append([0] * n_inputs)
        features_data.append(mfccs)
    features_data = np.array(features_data)
    return features_data

#获取训练集和对应标签
def get_train_set(train_labels): #参数为训练labels
    tr_data = fill_zero(train_features)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(tr_data))) #排列组合数据，len(tr_data)=总的样本语音条数=16*20=320
    x_shuffled = tr_data[shuffle_indices]
    y_shuffled = train_labels[shuffle_indices]
    #y_shuffled = one_hot_labels(y_shuffled)
    # 320个数据集切分为两部分，20%和80%。后20%~80%作为训练集共256个，前0~19作为测试集
    dev_sample_index = 1 * int(0.2 * float(len(y_shuffled))) # 0.2*320=64
    train_x, test_x = x_shuffled[dev_sample_index:], x_shuffled[:dev_sample_index]
    train_y, test_y = y_shuffled[dev_sample_index:], y_shuffled[:dev_sample_index]
    return train_x, train_y, test_x, test_y

def batch_iter(data,  batch_size,  num_epochs,  shuffle=True): #batch_size=16, num_epochs = 20000
    # 主要功能
    # 1 选择每次迭代,是否洗数据,像洗牌意义
    # 2 用生成器,每次只输出shuffled_data[start_index:end_index]这么多
    # Generates a batch iterator for a dataset.
    data = np.array(data)
    data_size = len(data)
    #epoch是所有训练数据的训练次数。一个完整的数据集通过神经网络一次并返回了一次的过程(正向传递+反向传递)称为一个epoch
    # 每个epoch的num_batch
    #单次epoch=（全部训练样本/batch_size） / iteration =1 ,即迭代的次数等于batch的数目

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1 #计算每个epoch有多少个batch个数

    print("num_batches_per_epoch:", num_batches_per_epoch)
    print("num_batches_per_epoch*num_epochs:", num_batches_per_epoch*num_epochs)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size,  data_size)
            #每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。
            yield shuffled_data[start_index:end_index]

#参数：批次，序列号（分帧的数量），每个序列的数据(batch, 待计算wav_max_len, 每帧的特征数)
with tf.name_scope('Inputs'):
    with tf.name_scope('Inputs_x'):
        x = tf.placeholder(tf.float32, [None, wav_max_len, n_inputs], name='Inputs_x')
    with tf.name_scope('Inputs_y'):
        y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout
# labels转one_hot格式
one_hot_labels = tf.one_hot(indices=tf.cast(y,  tf.int32),  depth=n_classes)


# X  input tensor的参数：batch_size 训练一个batch的图像数,
# in_hight 图片高度, in_width 图片宽度, in_channels 图像通道数
# W  filter的参数： filter_hight 卷积核高度, filter_width 卷积核宽度,
#  in_channels (就是input的第四维) 图像通道数，out_channels 卷积核个数
def conv2d(name, x, W, b, s=1):
    with tf.name_scope(name):
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME'))

def maxpool2d(name, x, k=2, s=2):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                              padding='SAME',name=name)

def norm(name, l_input, lsize=4):
    with tf.name_scope(name):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,
                     beta=0.75, name=name)


def conv_net(x, weights, biases, dropout):
    with tf.name_scope('conv_net'):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, wav_max_len, n_inputs, 1])  # 载入数据的大小 wav_max_len=46, n_inputs = 26 待测， 26
        # 输入的tensor x[16, 46, 26], batch_size, width, hight
        # the filter/kernel tensor wc1[3,3,1,64] , baise bc1 [64]
        # after conv2d ops, con1 [16*64, 46,26]
        conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'], s=1)
        tf.summary.histogram('wc1', weights['wc1'])
        tf.summary.histogram('bc1',biases['bc1'])
        tf.summary.histogram('covnv1', conv1)
        print('cov1.shape: ', conv1.get_shape().as_list())
        # max-pooling1, pool1 [64*16, 23, 13]
        pool1 = maxpool2d('pool1', conv1, k=2, s=2)
        print('pool1.shape: ', pool1.get_shape().as_list())
        norm1 = norm('norm1', pool1) # norm1 = [64*16, 23, 13]

        # after conv2d, [128*16, 23, 13]
        conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'], s=1)
        #max-pooling2 , [128*16, 12, 7]
        pool2 = maxpool2d('pool2', conv2, k=2, s=2)
        print('pool2.shape: ', pool2.get_shape().as_list())
        norm2 = norm('pool2', pool2) # norm2 = [128*16, 12, 7]

        fc1 = tf.reshape(norm2, [-1, weights['wd1'].get_shape().as_list()[0]]) # fc1 [16, 12*7*128]
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1']) # fc1 =[16, 1024], + bd1[1024]
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2']) # fc2= [16,1024], + bd2[1024]
        fc2 = tf.nn.relu(fc2)

        out = tf.matmul(fc2, weights['out']) + biases['out'] # w_out = [16， 16], +  b_out= [16]
        return out

with tf.name_scope('Weights'):
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),

        'wd1': tf.Variable(tf.random_normal([12 * 7 * 128, 1024])),
    # 第一个参数wav_max_len/(2*2) ,第二个参数7，为26连续两次max_pooling后，所得
        'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

with tf.name_scope('Bais'):
    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),

        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

with tf.name_scope('pred'):
    pred = conv_net(x, weights, biases, keep_prob)
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope('correct_pred'):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

#TF计算图
with tf.Session(config=config) as sess:
    train_x, train_y, test_x, test_y = get_train_set(train_labels)# 是个tuple的shape分别为（320*0.8=256，46,26）, (320*0.8=256,), (64,46,26),(64,)
    batches = batch_iter(list(zip(train_x, train_y)), batch_size, training_iters)
    batches = list(batches) #320000*16*16

    print (np.array(batches).shape) # (32000, 16, 16)；第三维的[0]代表 batch_x （特征数字）, [1]代表 batch_y（标签），均为 tuple 元组类型
    sess.run(init)
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter("logs", sess.graph) # 生成tensorboard 上的图
    step = 1
    for step, batch in enumerate(batches): # batch <class 'tuple'>:(16,2)
        step += 1
        batch_x, batch_y = zip(*batch) # shape(batch_x)=(16,46,26) , shape(batch_y)=(16,16)
        # 执行反向传播优化过程
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0: #当step累计到display_step时显示
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

            # summary = sess.run(merged, feed_dict={x:batch_x, y:batch_y})
            # train_writer.add_summary(summary, step)
            print("Iter: "+str(step) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))

    print("Optimization Finished!")

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.}))
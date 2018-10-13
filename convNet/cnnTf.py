import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pylab as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


array = [[2,3,4],
         [5,6,7],
         [8,9,0]]
print(array[0])

data = input_data.read_data_sets('data/fashion',one_hot=True)
# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))


# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

#print the max and min np.array of the image matrix
data.train.images[0]
# max = np.max(data.train.images[0])
# min = np.min(data.train.images[0])

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1) #为何是-1,the desired shape for your input layer : [batch_size, 28, 28, 1]
test_X = data.test.images.reshape(-1,28,28,1)
print("train_X.shape, test_X.shape=",train_X.shape, test_X.shape)

train_y = data.train.labels
test_y = data.train.labels

#hyperparameter setting
training_iters = 200
learning_rate = 0.001
batch_size = 128 # 通常是2 的幂次方，符合处理器的处理机制

# MNIST data input (img shape: 28*28)
n_input = 28*28
# MNIST total classes (0-9 digits)
n_classes = 10

#both placeholders are of type float
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # input tensor的参数：batch_size 训练一个batch的图像数, in_hight 图片高度, in_width 图片宽度, in_channels 图像通道数
    # filter的参数： filter_hight 卷积核高度, filter_width 卷积核宽度, in_channels (就是input的第四维) 图像通道数，out_channels 卷积核个数
    #padding的参数为 SAME（卷积核可到达边缘） 或 VALID（卷积核不能到达边缘）
    #https://blog.csdn.net/mao_xiao_feng/article/de    tails/53444333
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') #strides[0],strides[1]默认为1，中间两个代表卷积核往x,y 方向分别移动1
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')


weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}


def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    print("shape of Cov1:",conv1.get_shape())
    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    print("shape of Cov2:",conv2.get_shape())
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)  #conv3: 128*2028维

    print("shape of Cov3:",conv3.get_shape())
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 : 128*2048维
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]]) #weights['wd1'].get_shape().as_list()[0] = 4*4*128=2048
    #fc1: 128*128维
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases)
print("pred=",pred)
print("y=",y)




cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch_num in range(len(train_X)//batch_size):
            # print("batch=",batch,"type=",type(batch))
            # print("batch_size=",batch_size,"type=",type(batch_size))
            # print("len(train_X=",len(train_X),"type=",type(train_X))
            batch_x = train_X[batch_num*batch_size:min((batch_num+1)*batch_size,len(train_X))]
            batch_y = train_y[batch_num*batch_size:min((batch_num+1)*batch_size,len(train_y))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X, y: test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()



# def testTF():
#     # a = tf.constant(2.0)
#     # b = tf.constant(3.0)
#     # c = a*b
#     # sess = tf.Session()
#     # sess.run(c)
#
#     a = tf.placeholder(tf.float16)
#     b = tf.placeholder(tf.float16)
#
#     add= a+b
#     #creat a seesion obj
#     sess = tf.Session()
#     output= sess.run(add,{a:[1,2],b:[3,4]})
#     print("Adding a and b:", output)
#
#     #use variables
#     variable = tf.Variable([0.9,0.7],dtype=tf.float16)
#     #variable must be used before a graph is used for the first time
#     init = tf.global_variables_initializer()
#     sess.run(init)

#testTF()

#creating placeholders

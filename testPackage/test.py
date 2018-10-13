import tensorflow as tf
import numpy as np
# a = [[1,2,3],[5,6,7]]
# A = np.array(a)
# print(A.shape)
# print(int(3.5))
# train_labels = np.load('C:\\Users\\zkycs3\\Desktop\\SpeechCuishou1\\speechCNN\\train_8k_labels.npy')
#
# print(train_labels)


labels1 = [1, 3, 4, 8, 7, 5, 2, 9, 0, 8, 7,11,12,13,14,15]
labels2 = [1, 3, 4, 8, 7, 5, 2, 9, 0, 8, 7,11,12,13,14,15]
labels3 = [0]
labels4 = [1]

labels =[]
for i in np.arange(16):
    labels.append([i]*16)
    print(labels)

labels3.append([1]*10)
print(labels3)
for i in labels1:
    print(i)
labels = []
b = [121,131]
c = [99,100]
labels1.append(b)
labels.append(labels2)
labels.append(c)

# print(labels1)
# print(labels)
# one_hot_index = np.arange(len(labels)) * len(labels) + labels
#
# print('one_hot_index:{}'.format(one_hot_index))
#
# one_hot = np.zeros((len(labels), len(labels)))
# one_hot.flat[one_hot_index] = 1
#
# print('one_hot:{}'.format(one_hot))

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

# train_x, train_y = np.random.randint(0,10,size=[256,46,26]), np.random.randint(0,10,size=[16*16])
#
# batches = batch_iter(list(zip(train_x, train_y)), 16, 100)
# batches = list(batches)  # 320000*16*2
#
# for step, batch in enumerate(batches):  # batch <class 'tuple'>:(16,2)
#     step += 1
#     batch_x, batch_y = zip(*batch)  # shape(batch_x)=(16,46,26) , shape(batch_y)=(16)
#
# print(train_x)
# print(train_y)


# shuffle_indices = np.random.permutation(np.arange(320))
#
# print(shuffle_indices)
# features = np.load('C:\\Users\\zkycs3\\Desktop\\SpeechCuishou1\\train_8k_fbank_features.npy') #320*t*26维（语音条数，时长，丹帧特征数）
# features = np.array(features)
# print(features.shape)
# print(len(features))
# print(len(features[0][2])) #features[0]=27,features[0][2]=26
# print(len(features[20][3])) #features[20]=33,features[20][3]=26
# print(len(features[222])) #features[222]=19
# lables = np.load('C:\\Users\\zkycs3\\Desktop\\SpeechCuishou1\\train_8k_labels.npy')
# lables = np.array(lables)
# print(len(lables))
# print(lables)

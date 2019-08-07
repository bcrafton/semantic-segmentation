
# https://github.com/toimcio/SegNet-tensorflow/blob/master/layers_object.py
# how they did it.

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import keras
from gluoncv.data import CitySegmentation

'''
train_dataset = CitySegmentation(split='train')
val_dataset = CitySegmentation(split='val')

train_examples = len(train_dataset)
val_examples = len(val_dataset)
'''

'''
dataset = np.load('dataset.npy', allow_pickle=True).item()
x_train, y_train = dataset['x_val'], dataset['y_val']

print (np.shape(x_train))
print (np.shape(y_train))
'''

batch_size = 5
epochs = 10

####################################

def max_pool(inputs):
    value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.to_float(value), index, inputs.get_shape().as_list()

def up_sampling(pool, ind, output_shape, batch_size):
    pool_ = tf.reshape(pool, [-1])
    batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b = tf.reshape(b, [-1, 1])
    ind_ = tf.reshape(ind, [-1, 1])
    ind_ = tf.concat([b, ind_], 1)
    ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
    ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
    return ret

####################################

def encoder_block(x, filter_size, pool_size):
    conv1 = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn1   = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn2   = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(bn2)

    pool, idx, shape = max_pool(relu2)
    return pool, idx, shape

def decoder_block(x, idx, shape, filter_size, pool_size):
    unpool = up_sampling(x, idx, shape, batch_size)

    conv1 = tf.layers.conv2d(inputs=unpool, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn1   = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn2   = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(bn2)

    return relu2

####################################

x = tf.placeholder(tf.float32, [5, 480, 480, 3])
y = tf.placeholder(tf.int64, [5, 480, 480])
labels = tf.one_hot(y, depth=30, axis=-1)

encode1, idx1, shape1 = encoder_block(x,       64,  2)
encode2, idx2, shape2 = encoder_block(encode1, 128, 2)
encode3, idx3, shape3 = encoder_block(encode2, 256, 2)
encode4, idx4, shape4 = encoder_block(encode3, 512, 2)

# had issue with channel size in decode blocks.
# if corresponding output block output 256 channels, we need to give it back 256 channels.
# but with vgg you have 2 filters:
# 256, 512 and 512, 512
# so we were just saying num filters = 512 like the encoder case
# but for decoder this has to be different. 

decode1               = decoder_block(encode4, idx4, shape4, 256, 2)
decode2               = decoder_block(decode1, idx3, shape3, 128, 2)
decode3               = decoder_block(decode2, idx2, shape2, 64,  2)
decode4               = decoder_block(decode3, idx1, shape1, 30,  2)

out                   = decode4
predict               = tf.argmax(tf.nn.softmax(out), axis=3)

####################################

correct = tf.equal(predict, y)
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=out)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).minimize(loss)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(epochs):
    total_correct = 0 
    for jj in range(0, 500, batch_size):
        '''        
        s = jj
        e = jj + batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        '''
        xs = np.random.uniform(size=(5, 480, 480, 3))
        ys = np.random.uniform(size=(5, 480, 480))

        # [e1, e2, e3, e4] = sess.run([encode1, encode2, encode3, encode4, decode2], feed_dict={x: xs, y: ys})
        # print (np.shape(e1), np.shape(e2), np.shape(e3), np.shape(e4), shape1, shape2, shape3, shape4)

        # [d1, d2, d3, d4] = sess.run([decode1, decode2, decode3, decode4], feed_dict={x: xs, y: ys})        
        # print (np.shape(d1), np.shape(d2), np.shape(d3), np.shape(d4))

        [_sum_correct, _labels] = sess.run([sum_correct], feed_dict={x: xs, y: ys})
        total_correct += _sum_correct

    print (total_correct)





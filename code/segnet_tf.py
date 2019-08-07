
# https://github.com/toimcio/SegNet-tensorflow/blob/master/layers_object.py
# how they did it.

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

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

    pool, idx, shape = max_pool(inputs=relu2)
    return pool, idx, shape

def decoder_block(pool, idx, shape, filter_size, pool_size):
    conv1 = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn1   = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn2   = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(bn2)

    up = up_sampling(pool, idx, shape, batch_size)
    return up

####################################

x = tf.placeholder(tf.float32, [5, 480, 480, 3])
y = tf.placeholder(tf.int64, [5, 480, 480])

encode1, idx1, shape1 = encoder_block(x,       64,  2)
encode2, idx2, shape2 = encoder_block(encode1, 128, 2)
encode3, idx3, shape3 = encoder_block(encode2, 256, 2)
encode4, idx4, shape4 = encoder_block(encode3, 512, 2)

decode1               = decoder_block(encode4, idx4, shape4, 512, 2)
decode2               = decoder_block(decode1, idx3, shape3, 256, 2)
decode3               = decoder_block(decode2, idx2, shape2, 128, 2)
decode4               = decoder_block(decode3, idx1, shape1, 64,  2)

'''
out                   = tf.layers.conv2d(inputs=decode4, filters=30, kernel_size=[3, 3], padding='same')
predict               = tf.nn.softmax(out)
'''

####################################
'''
correct = tf.equal(tf.argmax(predict, axis=3), y)
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).minimize(loss)
'''
####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(epochs):
    for jj in range(0, 500, batch_size):
        s = jj
        e = jj + batch_size
        # xs = x_train[s:e]
        # ys = y_train[s:e]
        xs = np.random.uniform(size=(5, 480, 480, 3))
        ys = np.random.uniform(size=(5, 480, 480))
        [e4, d4] = sess.run([encode4, decode4], feed_dict={x: xs, y: ys})
        print (np.shape(e4), np.shape(d4))




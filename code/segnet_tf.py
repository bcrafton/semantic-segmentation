
# https://github.com/toimcio/SegNet-tensorflow/blob/master/layers_object.py
# how they did it.

import numpy as np
import tensorflow as tf
import keras
from gluoncv.data import CitySegmentation

train_dataset = CitySegmentation(split='train')
val_dataset = CitySegmentation(split='val')

train_examples = len(train_dataset)
val_examples = len(val_dataset)

batch_size = 5
epochs = 10

####################################

def encoder_block(x, filter_size, pool_size):
    conv1 = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn1   = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn2   = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(bn2)

    pool = tf.layers.max_pooling2d(inputs=relu2, pool_size=[pool_size, pool_size], strides=pool_size, padding='same')

    return pool

def decoder_block(x, filter_size, pool_size):
    conv1 = tf.layers.conv2d(inputs=x, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn1   = tf.layers.batch_normalization(conv1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.layers.conv2d(inputs=relu1, filters=filter_size, kernel_size=[3, 3], padding='same')
    bn2   = tf.layers.batch_normalization(conv2)
    relu2 = tf.nn.relu(bn2)

    up = tf.keras.layers.UpSampling2D() # tf.layers.max_pooling2d(inputs=relu2, pool_size=[pool_size, pool_size], strides=pool_size, padding='same')

    return pool

####################################

x = tf.placeholder(tf.float32, [None, 480, 480, 3])
y = tf.placeholder(tf.float32, [None, 480, 480])

encode1 = encoder_block(bn,      64,  2)
encode2 = encoder_block(encode1, 128, 2)
encode3 = encoder_block(encode2, 256, 2)
encode4 = encoder_block(encode3, 512, 2)

decode1 = decoder_block(encode6, 512, 2)
decode2 = decoder_block(decode1, 256, 2)
decode3 = decoder_block(decode2, 128, 2)
decode4 = decoder_block(decode3, 64,  2)

out = tf.softmax(decode4)

####################################

predict = tf.argmax(out, axis=3)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).minimize(loss)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(epochs):
    for jj in range(0, train_examples, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([train], feed_dict={x: xs, y: ys})
        
    total_correct = 0
    for jj in range(0, val_examples, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_test[s:e]
        ys = y_test[s:e]
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct
            
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        

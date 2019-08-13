
####################################

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import mxnet as mx

import tensorflow as tf
import keras

from gluoncv.data import CitySegmentation

from lib.SegNet import SegNet

####################################

# train_dataset = CitySegmentation(split='train')
# train_examples = len(train_dataset)

val_dataset = CitySegmentation(split='val')
val_examples = len(val_dataset)

batch_size = 1
epochs = 10

####################################

image         = tf.placeholder(tf.float32, [batch_size, 480, 480, 3])
label         = tf.placeholder(tf.int64, [batch_size, 480, 480])
label_one_hot = tf.one_hot(label, depth=30, axis=-1)

model   = SegNet(batch_size=batch_size, init='glorot_uniform', load='/usr/scratch/bcrafton/cityscapes/code/MobileNetWeights.npy')
out     = model.predict(image)
predict = tf.argmax(tf.nn.softmax(out), axis=3)

####################################

correct = tf.equal(predict, label)
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(epochs):

    total_correct = 0 
    for jj in range(0, 500, batch_size):
        print (jj)
        x, y = val_dataset[jj]
        xs = x.asnumpy().reshape([1, 480, 480, 3])
        ys = y.asnumpy().reshape([1, 480, 480])
        [_sum_correct] = sess.run([sum_correct], feed_dict={image: xs, label: ys})
        total_correct += _sum_correct

    print (total_correct)

####################################



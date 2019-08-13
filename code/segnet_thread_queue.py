
####################################

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import mxnet as mx
import tensorflow as tf
import keras
import queue
import threading

from gluoncv.data import CitySegmentation

from lib.SegNet import SegNet

####################################

train_dataset = CitySegmentation(split='train')
train_examples = len(train_dataset)

val_dataset = CitySegmentation(split='val')
val_examples = len(val_dataset)

batch_size = 5
epochs = 10

train_dataset_np = train_dataset.asnumpy()

####################################

def fill_queue(d, q):
    ii = 0

    while(True):
        if q.full() == False:
            xs = []
            ys = []

            for jj in range(batch_size):
                x, y = d[ii + jj]
                xs.append(x.asnumpy())
                ys.append(y.asnumpy())

            xs = np.stack(xs, axis=0)
            ys = np.stack(ys, axis=0)
            q.put((xs, ys))

            ii = (ii + batch_size) if (ii < train_examples) else 0
            print (q.qsize())

####################################

q = queue.Queue(maxsize=10)
thread = threading.Thread(target=fill_queue, args=(train_dataset, q))
thread.start()

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

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_one_hot, logits=out)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).minimize(loss)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(epochs):

    total_correct = 0.
    total_labels = 0.
    losses = []
    for jj in range(0, train_examples, batch_size):

        xs, ys = q.get()
        [_sum_correct, _loss, _] = sess.run([sum_correct, loss, train], feed_dict={image: xs, label: ys})

        total_correct += _sum_correct
        total_labels += batch_size * 480 * 480
        losses.append(_loss)        
        
        print ('%d %f %f' % (jj, total_correct / total_labels, np.average(losses)))

####################################



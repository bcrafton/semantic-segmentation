
####################################

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import keras
from gluoncv.data import CitySegmentation

from lib.SegNet import SegNet

####################################

# train_dataset = CitySegmentation(split='train')
# train_examples = len(train_dataset)

val_dataset = CitySegmentation(split='val')
val_examples = len(val_dataset)

batch_size = 5
epochs = 10

####################################

image         = tf.placeholder(tf.float32, [5, 480, 480, 3])
label         = tf.placeholder(tf.int64, [5, 480, 480])
label_one_hot = tf.one_hot(label, depth=30, axis=-1)

model   = SegNet(batch_size=batch_size, init='glorot_uniform')
out     = model.predict(image)
predict = tf.argmax(tf.nn.softmax(out), axis=3)

####################################

correct = tf.equal(predict, label_one_hot)
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=out)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).minimize(loss)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(epochs):

    total_correct = 0 
    for jj in range(0, 500, batch_size):

        xs = []; ys = []
        for kk in range(batch_size):
            x, y = val_dataset[jj + kk]
            xs.append(x)
            ys.append(y)
        xs = np.stack(xs, axis=0)
        ys = np.stack(ys, axis=0)

        [_sum_correct] = sess.run([sum_correct], feed_dict={image: xs, label: ys})
        total_correct += _sum_correct

    print (total_correct)

####################################



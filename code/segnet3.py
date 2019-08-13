
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="segnet")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import numpy as np
import tensorflow as tf
import keras
from lib.SegNet import SegNet

####################################

train_path = '/usr/scratch/bcrafton/cityscapes/train'
val_path = '/usr/scratch/bcrafton/cityscapes/val'

####################################

def get_val_filenames():
    val_filenames = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            val_filenames.append(os.path.join(val_path, file))

    np.random.shuffle(val_filenames)    

    return val_filenames
    
def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            train_filenames.append(os.path.join(train_path, file))
    
    np.random.shuffle(train_filenames)

    return train_filenames

def extract_fn(record):
    _feature={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)
    }

    sample = tf.parse_single_example(record, _feature)

    image = tf.decode_raw(sample['image_raw'], tf.float32)
    image = tf.reshape(image, (1, 480, 480, 3))

    label = tf.decode_raw(sample['label_raw'], tf.int32)
    label = tf.reshape(label, (1, 480, 480))

    return [image, label]

####################################

train_filenames = get_train_filenames()
val_filenames = get_val_filenames()

train_examples = len(train_filenames)
val_examples = len(val_filenames)

####################################

filename = tf.placeholder(tf.string, shape=[None])
batch_size = tf.placeholder(tf.int32, shape=())
lr = tf.placeholder(tf.float32, shape=())

####################################

val_dataset = tf.data.TFRecordDataset(filename)
val_dataset = val_dataset.map(extract_fn, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

train_dataset = tf.data.TFRecordDataset(filename)
train_dataset = train_dataset.map(extract_fn, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

####################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
x, y = iterator.get_next()

image         = tf.reshape(x, [batch_size, 480, 480, 3])
label         = tf.reshape(y, [batch_size, 480, 480])
label_one_hot = tf.one_hot(label, depth=30, axis=-1)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

####################################

model   = SegNet(batch_size=batch_size, init='glorot_uniform', load='/usr/scratch/bcrafton/semantic-segmentation/code/MobileNetWeights.npy')
out     = model.predict(image)
predict = tf.argmax(tf.nn.softmax(out), axis=3)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################

for ii in range(args.epochs):
    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})

    for jj in range(0, train_examples, args.batch_size):
        [pred] = sess.run([predict], feed_dict={handle: train_handle, batch_size: args.batch_size, lr: args.lr})

####################################



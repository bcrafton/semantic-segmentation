
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ReLU, BatchNormalization, Softmax, UpSampling2D
from gluoncv.data import CitySegmentation

train_dataset = CitySegmentation(split='train')
val_dataset = CitySegmentation(split='val')

train_examples = len(train_dataset)
val_examples = len(val_dataset)

#####################

def model_add(model, layers):
    for l in layers:
        model.add(l)

#####################

'''
x_train, y_train = zip(*train_dataset)
x_val, y_val = zip(*val_dataset)
'''

'''
count = 0
x_train = []; y_train = []
for (x, y) in train_dataset:
    print (count)
    count = count + 1
    x_train.append(x); y_train.append(y)

count = 0
x_val = []; y_val = []
for (x, y) in val_dataset:
    print (count)
    count = count + 1
    x_val.append(x); y_val.append(y)

dataset = {}
dataset['x_train'] = x_train
dataset['y_train'] = y_train
dataset['x_val'] = x_val
dataset['y_val'] = y_val
np.save('dataset', dataset)
'''

dataset = np.load('dataset.npy').item()
x_val, y_val = dataset['x_val'], dataset['y_val']

print ('loaded dataset')

#####################

batch_size = 5
epochs = 10

####################################

def encoder_block(filter_size, pool_size, input_shape=None):
    if input_shape is not None:
        conv1 = Conv2D(filter_size, kernel_size=[3, 3], padding="same", use_bias=False, input_shape=input_shape)
    else:
        conv1 = Conv2D(filter_size, kernel_size=[3, 3], padding="same", use_bias=False)

    bn1   = BatchNormalization()
    relu1 = ReLU()

    conv2 = Conv2D(filter_size, kernel_size=[3, 3], padding="same", use_bias=False)
    bn2   = BatchNormalization()
    relu2 = ReLU()

    pool = MaxPooling2D(pool_size=2, strides=2, padding='same')

    return [conv1, bn1, relu1, conv2, bn2, relu2, pool]

def decoder_block(filter_size, pool_size):
    conv1 = Conv2D(filter_size, kernel_size=[3, 3], padding="same", use_bias=False)
    bn1   = BatchNormalization()
    relu1 = ReLU()

    conv2 = Conv2D(filter_size, kernel_size=[3, 3], padding="same", use_bias=False)
    bn2   = BatchNormalization()
    relu2 = ReLU()

    pool = UpSampling2D(size=2)

    return [conv1, bn1, relu1, conv2, bn2, relu2, pool]

####################################

model = Sequential()

model_add(model, encoder_block(64,  2))
model_add(model, encoder_block(128, 2))
model_add(model, encoder_block(256, 2))
model_add(model, encoder_block(512, 2))

model_add(model, decoder_block(512, 2))
model_add(model, decoder_block(256, 2))
model_add(model, decoder_block(128, 2))
model_add(model, decoder_block(64,  2))
model_add(model, decoder_block(30,  1))

model.add(Softmax()) 

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_val, y_val, batch_size=4, epochs=1, verbose=0, validation_data=[x_val, y_val])
        
        

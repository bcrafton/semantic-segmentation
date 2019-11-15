
from gluoncv.data import CitySegmentation
import matplotlib.pyplot as plt
import numpy as np

#########################

train_dataset = CitySegmentation(split='train')
val_dataset = CitySegmentation(split='val')

print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

#########################

for ii in range(100):
    print (ii)
    img, mask = val_dataset[ii]
    img = img.asnumpy()
    img = img / np.max(img)
    plt.imsave('%d.jpg' % (ii), img)

#########################


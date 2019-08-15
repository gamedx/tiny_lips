import os
import scipy
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
%matplotlib inline

from PIL import Image  # used to read images from directory


#%%
## read in files

path = os.path.join('data', 'mnist')
fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainX = loaded[16:].reshape((8400, 65, 65, 1)).astype(np.float32)

fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
loaded = np.fromfile(file=fd, dtype=np.uint8)
trainY = loaded[8:].reshape((8400)).astype(np.int32)

trX = trainX[:7000] / 255.
trY = trainY[:7000] 

valX = trainX[7000:, ] / 255.
valY = trainY[7000:]

#%%

# the tfrecord file path, you need to create the folder yourself
recordPath = "./tfrecord/"

# number of images stored in each tfrecord file
bestNum = 1000
recordFileNum = 0

 
recordFileName = ("train.tfrecords-%.3d" % recordFileNum)
# tfrecord file writer
writer = tf.io.TFRecordWriter(recordPath + recordFileName)

print("Creating the 000 tfrecord file")

for idx in range(trainX.shape[0]):
    if (idx % bestNum == 0) and (idx != 0):
        recordFileNum += 1
        recordFileName = ("train.tfrecords-%.3d" % recordFileNum)
        writer = tf.io.TFRecordWriter(recordPath + recordFileName)
        print("Creating the %.3d tfrecord file" % recordFileNum)
    xInBytes = trainX[idx].tobytes()
    label = trainY[idx]
    example = tf.train.Example(features=tf.train.Features(feature={"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[xInBytes])), "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
    writer.write(example.SerializeToString())


writer.close()

#%%


plt.imshow(np.squeeze(trainX[1] / 255, axis=2), interpolation='nearest', cmap='gray')
plt.show()


import os
import numpy as np
import pandas as pd
import tensorflow as tf

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

# 生成浮点型的属性
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

file_path = "D:/CNN_INPUT_TEST"
files = next(os.walk(file_path))[2]

sample_num = 100
win_size = 9
cova_num = 16

images = np.ndarray(shape = (sample_num, 16, 81), dtype = np.float64)
index = 0;
for file in files:
    filename = os.path.join(file_path, file)
    window = pd.read_table(filename, sep='\t').values
    window = window.T
    images[index, ] = window
    index = index + 1

labels = pd.read_table("D:/labels.txt", sep='\t').values

filename = "D:/output.tfrecords"
writer = tf.python_io.TFRecordWriter(filename)
for i in range(sample_num):
    image_raw = images[i].tostring()
    example = tf.train.Example(features = tf.train.Features(feature = {
        'label': _float_feature(labels[i]),
        'image_raw': _bytes_feature(image_raw),
        'height': _int64_feature(win_size),
        'width': _int64_feature(win_size),
        'channels': _int64_feature(cova_num)
    }))
    writer.write(example.SerializeToString())
writer.close()

print('end')
    

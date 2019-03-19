
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


class CDCollection:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.batch_index = 0
        self.sample_num = images.shape[0]
    def next_batch(batch_size):
        start = self.batch_index
        end = start + batch_size
        self.batch_index += batch_size
        return self.images[index:batch_size], self.labels[index:batch_size] 

win_size = 9
cova_num = 16

file_path = "D:/CNN_ALL_9/"
files = next(os.walk(file_path))[2]
sample_num = len(files)
images = np.ndarray(shape = (sample_num, win_size,win_size, cova_num), dtype = np.float64)
images_add = np.ndarray(shape = (sample_num * 3, 9, 9, 16), dtype = np.float64)
labels_data = pd.read_table("D:/labels.txt", sep='\t').values
id_arr = labels_data[:, 0]
labels = list()
index = 0
min_max_scaler = preprocessing.MinMaxScaler()
for file in files:
    filename = file_path + file
    df = pd.read_table(filename, sep='\t').values
    ## 归一化处理
    df = min_max_scaler.fit_transform(df)

    image = df.reshape(win_size, win_size, cova_num)
    images[index, ] = image

    index = index + 1
    # 加载labels
    id = file.split('_')[0].split('.')[0]
    ind = np.where(id_arr == int(id))
    dtb = labels_data[ind, 1]
    if(dtb.shape[1] == 0):
        print('error')
    labels.append(dtb[0, 0])

labels = np.array(labels)
cc = CDCollection(images, labels)

import pickle
fw = open("D:/train_data_all_9_scale", "wb")
pickle.dump(cc, fw)
fw.close()

'''
fr = open("D:/train_data", "rb")
cc = pickle.load(fr)
fr.close()
'''

print('end')
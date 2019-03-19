import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
import pickle

#  加载dsm_inference.py和dsm_train.py 中定义的常量和函数
import dsm_inference
import dsm_train

class CDCollection:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.batch_index = 0
        self.sample_num = images.shape[0]
    def next_batch(self, batch_size):
        start = self.batch_index
        end = start + batch_size
        flag = False
        if(end >= self.sample_num):
            flag = True
        self.batch_index += batch_size
        self.batch_index = self.batch_index % self.sample_num
        if flag == False:
            return self.images[start:end,], self.labels[start:end]
        else:
            return np.append(self.images[start:self.sample_num,], self.images[0:(start+batch_size-self.sample_num),], axis = 0), np.append(self.labels[start:self.sample_num], self.labels[0:(start+batch_size-self.sample_num)])


# 每10s加载一次新的模型，并在测试数据上测试新模型的正确性
EAVL_INTERVAL_SECS = 10
def evaluate(train_data):
    with tf.Graph().as_default() as g:
        print(train_data.sample_num)
        x = tf.placeholder(tf.float64, [train_data.sample_num, dsm_inference.IMAGE_SIZE, dsm_inference.IMAGE_SIZE, dsm_inference.NUM_CHANNELS], name="x-input")
        
        y_ = tf.placeholder(tf.float64, [train_data.sample_num, dsm_inference.OUTPUT_NODE], name = 'y-input')

        # x_images = np.reshape(dsm.validation.images, (dsm.validation.num_examples, dsm_inference.IMAGE_SIZE, dsm_inference.IMAGE_SIZE, dsm_inference.NUM_CHANNELS))
        ys  = np.reshape(train_data.labels, [train_data.sample_num,dsm_inference.OUTPUT_NODE])
        validate_feed = {x: train_data.images, y_: ys}

        y = dsm_inference.inference(x, False, None)

        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均模型来获取平均值了。
        variable_averages = tf.train.ExponentialMovingAverage(dsm_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state 函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(dsm_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型：
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return 
            time.sleep(EAVL_INTERVAL_SECS)

def main(argv=None):

    fr = open("D:/train_data_all_9_scale", "rb")
    cc = pickle.load(fr)
    fr.close()

    evaluate(cc)

if __name__ == "__main__":
    tf.app.run()
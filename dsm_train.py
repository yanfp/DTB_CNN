import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import numpy as np
import tensorflow as tf
import pickle

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
# 加载dsm_inference.py中定义的常量和前向传播函数
import dsm_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 80000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径及文件名
MODEL_SAVE_PATH = "D:/CNN_MODEL/"
MODEL_NAME = "model.ckpt"

import tensorflow as tf


def train(train_data):
    x = tf.placeholder(tf.float64, [BATCH_SIZE, dsm_inference.IMAGE_SIZE, dsm_inference.IMAGE_SIZE, dsm_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(tf.float64, [BATCH_SIZE, dsm_inference.OUTPUT_NODE], name = "y-input")

    #x = np.reshape(x, newshape=(batch_size, dsm_inference.IMAGE_SIZE, dsm_inference.IMAGE_SIZE))

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用dsm_inference.py中的前向传播过程
    y = dsm_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable = False)

    # 定义损失函数，学习率， 滑动平均操作以及训练过程
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_average_op = variable_average.apply(tf.trainable_variables())

    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    error = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))

    #cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = error + tf.add_n(tf.get_collection("losses"))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, train_data.sample_num / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name = 'train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化变量
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        for i  in range(TRAINING_STEPS):
            xs, ys = train_data.next_batch(BATCH_SIZE)
            #reshaped_xs = np.reshape(xs, (BATCH_SIZE, dsm_inference.IMAGE_SIZE, dsm_inference.IMAGE_SIZE, dsm_inference.NUM_CHANNELS))
            ys  = np.reshape(ys, [BATCH_SIZE,dsm_inference.OUTPUT_NODE])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_: ys})
            yv, y_v = sess.run([y, y_], feed_dict = {x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv = None):
    # 加载train_data数据
    fr = open("D:/train_data_all_9_scale", "rb")
    cc = pickle.load(fr)
    fr.close()

    train(cc)

if __name__ == '__main__':
    tf.app.run()



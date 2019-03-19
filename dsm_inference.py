
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

#配置神经网络的参数
INPUT_NODE = 9 * 9
OUTPUT_NODE = 1

IMAGE_SIZE = 9
NUM_CHANNELS = 16
NUM_LABELS = 1

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 16
CONV1_SIZE = 3
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 32
CONV2_SIZE = 3
# 全连接层的节点个数
FC_SIZE1 = 10
FC_SIZE2 = 10

# 定义卷积神经网络的前向传播过程
def inference(input_tensor, train, regularizer):
    # 实现第一层卷积层的前向传播过程
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1), dtype = tf.float64)
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0), dtype = tf.float64)

        # 使用边长为5，深度为32的过滤器，过滤器移动步长为1， 且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides = [1, 1, 1, 1], padding = "SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 实现第三层dropout(0.3)pool1
    dot1 = tf.nn.dropout(pool1, 0.3)

    # 实现第四层卷积层前向传播过程
    with tf.variable_scope("layer4-conv2"):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1), dtype = tf.float64)
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer = tf.constant_initializer(0.0), dtype = tf.float64)

        conv2 = tf.nn.conv2d(dot1, conv2_weights, strides = [1, 1, 1, 1], padding = "SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    # 实现第五层全连接前向传播过程
    relu_shape = relu2.get_shape().as_list()
    nodes = relu_shape[1] * relu_shape[2] * relu_shape[3]
    reshaped = tf.reshape(relu2, [relu_shape[0], nodes], name = "flatten")
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE1], initializer = tf.truncated_normal_initializer(stddev=0.1), dtype = tf.float64)
        # 只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE1], initializer=tf.constant_initializer(0.1), dtype = tf.float64)
        
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    # 实现第六层dropout(0.3)
    dot2 = tf.nn.dropout(fc1, 0.3)

    # 实现第七层全连接
    with tf.variable_scope("layer7-fc2"):
        fc2_weights = tf.get_variable('weight', [FC_SIZE1, FC_SIZE2], initializer=tf.truncated_normal_initializer(stddev=0.1), dtype = tf.float64)

        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [FC_SIZE2], initializer = tf.constant_initializer(0.1), dtype = tf.float64)
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases

    # 实现第八层输出层
    with tf.variable_scope("layer8-fc2"):
        fc3_weights = tf.get_variable('weight', [FC_SIZE2, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1), dtype = tf.float64)

        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [NUM_LABELS], initializer = tf.constant_initializer(0.1), dtype = tf.float64)
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
    return logit
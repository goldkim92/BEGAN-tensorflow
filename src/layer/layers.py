import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def conv2d(x, filter_shape, bias=True, stride=1, padding="SAME", name="conv2d"):
    kw, kh, nin, nout = filter_shape
    pad_size = (kw - 1) / 2

    if padding == "VALID":
        x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "SYMMETRIC")

    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=filter_shape, initializer=initializer)
        x = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding=padding)

        if bias:
            b = tf.get_variable("bias", shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
    return x


def fc(x, output_shape, bias=True, name='fc'):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(x, [-1, dim])
    input_shape = dim

    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=[input_shape, output_shape], initializer=initializer)
        x = tf.matmul(x, weight)

        if bias:
            b = tf.get_variable("bias", shape=[output_shape], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
    return x


def pool(x, r=2, s=1):
    return tf.nn.avg_pool(x, ksize=[1, r, r, 1], strides=[1, s, s, 1], padding="SAME")


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


#%% JM
def slim_conv2d(input_, output_dim, ks=3,s=1,padding='SAME',name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                           activation_fn=tf.nn.elu,
                           weights_initializer=tf.contrib.layers.xavier_initializer())
     
        
def slim_maxpool2d(input_, ks=2, s=2, padding='VALID',name='maxpool2d'):
    with tf.variable_scope(name):
        return slim.max_pool2d(input_, ks, s, padding=padding)

    
def slim_fully_connected(input_, num_outputs, name='fc'):
    with tf.variable_scope(name):
        return slim.fully_connected(input_, num_outputs,
                                    weights_initializer=tf.contrib.layers.xavier_initializer())
        
def slim_denseblock(input_, nk, reuse=False, name='denseblock'):   
    with tf.variable_scope(name):
        h_conv1 = slim_conv2d(input_, nk, name='h_conv1')
        h_conv2 = slim_conv2d(tf.concat((input_,h_conv1),axis=3), nk, name='h_conv2')
        h_conv3 = slim_conv2d(tf.concat((input_,h_conv1,h_conv2),axis=3), nk, name='h_conv3')
        h_conv4 = slim_conv2d(tf.concat((input_,h_conv1,h_conv2,h_conv3),axis=3), nk, name='h_conv4')
        h_conv5 = slim_conv2d(tf.concat((input_,h_conv1,h_conv2,h_conv3,h_conv4),axis=3), nk, name='h_conv5')
        
        return h_conv5
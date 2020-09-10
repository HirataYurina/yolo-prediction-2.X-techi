# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras


def saved_model_reshape():
    """I think maybe it is tf.squeeze and tf.reshape causing the bug.
    So, i saved a model with tf.squeeze and tf.reshape op.
    I recommend not to use tf.squeeze op in GPU environment.

    """
    inputs = keras.Input(shape=(13, 13, 3*4))
    x = tf.squeeze(inputs)
    shape = tf.shape(x)
    h = shape[0]
    w = shape[1]
    x = tf.reshape(x, shape=(h, w, 3, -1))

    model = keras.Model(inputs, x)
    tf.saved_model.save(model, export_dir='./saved_model_reshape')


if __name__ == '__main__':

    # i found it's ok to use tf.squeeze and tf.reshape together
    model_ = tf.saved_model.load('./saved_model_reshape')
    inputs_ = tf.ones(shape=(1, 13, 13, 3*4))

    x_ = model_.signatures['serving_default'](inputs_)
    print(x_)

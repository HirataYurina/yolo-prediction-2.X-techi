# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:predict.py
# software: PyCharm

import tensorflow as tf
from PIL import Image
import numpy as np
from nets.utils import letterbox_image

yolo4 = tf.saved_model.load('./saved_model')

image = Image.open('./images/danger_source.jpg')
image_shape = np.array([image.size[1], image.size[0]], dtype='float32')
image = letterbox_image(image, (416, 416))
image_array = np.expand_dims(np.array(image, dtype='float32') / 255.0, axis=0)
image_constant = tf.constant(image_array, dtype=tf.float32)
image_shape = tf.constant(image_shape, dtype=tf.float32)

# ---------------------------------------------------------------------------------------
# if you have multi-inputs of your model, you can not use positional argument rather than
# use key-word argument.
result = yolo4.signatures['serving_default'](input_2=image_constant, input_3=image_shape)
print(result)

# yolo4_ = tf.keras.models.load_model('./saved_model')
# boxes, scores, classes = yolo4([image_array, image_shape])
# print(boxes, scores, classes)

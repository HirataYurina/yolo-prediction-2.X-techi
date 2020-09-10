from functools import wraps

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from nets.darknet53 import csp_darknet_body
from nets.utils import compose


# --------------------------------------------------
# convolution without BN and activation
# --------------------------------------------------
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------
# Convolution + BatchNormalization + LeakyReLU
# ---------------------------------------------------
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# --------------------------------------------------------------------------
# feature map ---> outputs [batch, 13, 13, num_anchors*(num_classes+5)]
# This function is used in yoloV3
# --------------------------------------------------------------------------
def make_last_layers(x, num_filters, out_filters):

    # process 5 convolution operation
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # adjust the output channels to num_anchors*(num_classes+5)
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)
            
    return x, y


# ---------------------------------------------------------------------------------------
# CSPDarknet + SPP + PANet(yoloV4 changes adding operation to concatenating operation)
# TODO: Path Aggregation Network for Instance Segmentation
# https://arxiv.org/abs/1803.01534
#      |
#      |
#   -------        -------        -------
#   |     | -----> |     | -----> |     | -----> [batch, 52, 52, 3 * 85]
#   -------        -------        -------
#      |              ^              |
#      |              |              V
#   -------        -------        -------
#   |     | -----> |     | -----> |     | -----> [batch, 26, 26, 3 * 85]
#   -------        -------        -------
#      |              ^              |
#      |              |              V
#   -------        -------        -------
#   |     | -----> |     | -----> |     | -----> [batch, 13, 13, 3 * 85]
#   -------        -------        -------
#
# ---------------------------------------------------------------------------------------
def yolo_body(inputs, num_anchors, num_classes):

    # get feature map from backbone
    feat1, feat2, feat3 = csp_darknet_body(inputs)
    feat3 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    feat3 = DarknetConv2D_BN_Leaky(1024, (3, 3))(feat3)
    feat3 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)

    # spatial pooling pyramid
    # enhance respective fields
    pool1 = MaxPooling2D(13, strides=1, padding='same')(feat3)
    pool2 = MaxPooling2D(9, strides=1, padding='same')(feat3)
    pool3 = MaxPooling2D(5, strides=1, padding='same')(feat3)
    pool_fusion = Concatenate()([pool1, pool2, pool3, feat3])

    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(pool_fusion)
    y5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)
    y5_upsample = DarknetConv2D_BN_Leaky(256, (1, 1))(y5)
    y5_upsample = UpSampling2D()(y5_upsample)

    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    y4 = Concatenate()([y4, y5_upsample])
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)

    y4_upsample = DarknetConv2D_BN_Leaky(128, (1, 1))(y4)
    y4_upsample = UpSampling2D()(y4_upsample)

    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    y3 = Concatenate()([y3, y4_upsample])
    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)
    y3 = DarknetConv2D_BN_Leaky(256, (3, 3))(y3)
    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)
    y3 = DarknetConv2D_BN_Leaky(256, (3, 3))(y3)
    y3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)

    # the output of C3 and it reference to little anchors
    y3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(y3)
    y3_output = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1))(y3_output)

    # down sample and start aggregating the path from lower layers to higher layers
    # PANet
    y3 = ZeroPadding2D(padding=((1, 0), (1, 0)))(y3)
    y3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(y3)
    y4 = Concatenate()([y3_downsample, y4])
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)
    y4 = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4 = DarknetConv2D_BN_Leaky(256, (1, 1))(y4)

    # the output of C4 and it reference to middle anchors
    y4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(y4)
    y4_output = DarknetConv2D(num_anchors * (5 + num_classes), (1, 1))(y4_output)

    y4 = ZeroPadding2D(padding=((1, 0), (1, 0)))(y4)
    y4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(y4)
    y5 = Concatenate()([y4_downsample, y5])
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)
    y5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)
    y5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5 = DarknetConv2D_BN_Leaky(512, (1, 1))(y5)

    # the output of C5 and it reference to large anchors
    y5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(y5)
    y5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(y5_output)

    return Model(inputs, [y5_output, y4_output, y3_output])


if __name__ == '__main__':

    yolo = yolo_body(keras.Input(shape=(416, 416, 3)), 3, 80)
    yolo.summary()

    # yolo.load_weights('../logs/yolo4_weight.h5')
    #
    # for layer in yolo.layers[-3:]:
    #     print(layer.name)
    # print(len(yolo.layers))

    # from PIL import Image
    # from utils.utils import letterbox_image
    # import numpy as np

    # street = Image.open('../img/street.jpg')
    # print(street.size)
    #
    # street = letterbox_image(street, (416, 416))
    # street = np.expand_dims(np.array(street), axis=0)
    # street = street / 255.0
    # # street = tf.convert_to_tensor(street, dtype=tf.float32)
    #
    # with K.get_session() as sess:
    #     print(sess.run(yolo.output, {yolo.input: street, K.learning_phase(): 0}))

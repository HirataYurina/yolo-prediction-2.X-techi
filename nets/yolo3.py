from functools import wraps

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from nets.darknet53 import darknet_body
from nets.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def make_last_layers(x, num_filters, out_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    y = DarknetConv2D_BN_Leaky(num_filters*2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)
            
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    feat1, feat2, feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, feat2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, feat1])

    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])


if __name__ == '__main__':
    inputs = keras.Input(shape=(416, 416, 3))
    model = yolo_body(inputs, 3, 2)
    model.summary()

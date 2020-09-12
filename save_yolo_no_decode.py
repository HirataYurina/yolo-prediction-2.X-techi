# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:predict.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
from nets.yolo4 import yolo_body as yolo4_body
from nets.yolo3 import yolo_body as yolo3_body
from argparse import ArgumentParser


def arg():
    parser = ArgumentParser(description='save your model with saved-model format '
                                        'that does not contain decoding process')
    parser.add_argument('--num_anchors', type=int, default=3, help='number of your anchors')
    parser.add_argument('--num_classes', type=int, default=6, help='number of your classes')
    parser.add_argument('--weight_path', type=str, default='./data/ep290-loss16.428.h5',
                        help='the path of model weights')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="yolo4")
    args = parser.parse_args()

    return args


def main():
    # the arguments
    args = arg()
    num_anchors = args.num_anchors
    num_classes = args.num_classes
    weight_path = args.weight_path
    save_path = args.save_path
    model_name = args.model_name

    inputs_keras = keras.Input(shape=(416, 416, 3))

    assert model_name == 'yolo4' or model_name == 'yolo3', 'model must be yolo4 or yolo3'
    if model_name == 'yolo4':
        model = yolo4_body(inputs=inputs_keras,
                           num_anchors=num_anchors,
                           num_classes=num_classes)
    else:
        model = yolo3_body(inputs=inputs_keras,
                           num_classes=num_classes,
                           num_anchors=num_anchors)
    model.load_weights(weight_path)
    tf.saved_model.save(model, save_path)


if __name__ == '__main__':
    main()

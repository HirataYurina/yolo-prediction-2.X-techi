# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:predict.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
from nets.yolo4 import yolo_body
import numpy as np
from PIL import Image
from nets.utils import letterbox_image
from argparse import ArgumentParser


def arg():
    parser = ArgumentParser(description='save your model with saved-model format')
    parser.add_argument('--num_anchors', type=int, default=3, help='number of your anchors')
    parser.add_argument('--num_classes', type=int, default=6, help='number of your classes')
    parser.add_argument('--score', type=float, default=0.3, help='score threshold of prediction')
    parser.add_argument('--iou', type=float, default=0.5, help='iou threshold of prediction')
    parser.add_argument('--img_path', type=str, default='./images/danger_source.jpg',
                        help='image that you want to predict')
    parser.add_argument('--weight_path', type=str, default='./data/ep290-loss16.428.h5',
                        help='the path of model weights')
    parser.add_argument('--save_path', type=str, default='./saved_model')
    args = parser.parse_args()

    return args


# the arguments
args = arg()
num_anchors = args.num_anchors
num_classes = args.num_classes
score = args.score
iou = args.iou
img_path = args.img_path
weight_path = args.weight_path
save_path = args.save_path


anchor_xy = []

anchor_x_13, anchor_y_13 = np.meshgrid(np.arange(13), np.arange(13))
anchor_xy_13 = np.stack([anchor_x_13, anchor_y_13], axis=-1)
anchor_xy_13 = np.tile(np.reshape(anchor_xy_13, newshape=(13, 13, 1, 2)), (1, 1, 3, 1))
anchor_xy_13 = tf.constant(anchor_xy_13, dtype=tf.float32)
anchor_xy.append(anchor_xy_13)

anchor_x_26, anchor_y_26 = np.meshgrid(np.arange(26), np.arange(26))
anchor_xy_26 = np.stack([anchor_x_26, anchor_y_26], axis=-1)
anchor_xy_26 = np.tile(np.reshape(anchor_xy_26, newshape=(26, 26, 1, 2)), (1, 1, 3, 1))
anchor_xy_26 = tf.constant(anchor_xy_26, dtype=tf.float32)
anchor_xy.append(anchor_xy_26)

anchor_x_52, anchor_y_52 = np.meshgrid(np.arange(52), np.arange(52))
anchor_xy_52 = np.stack([anchor_x_52, anchor_y_52], axis=-1)
anchor_xy_52 = np.tile(np.reshape(anchor_xy_52, newshape=(52, 52, 1, 2)), (1, 1, 3, 1))
anchor_xy_52 = tf.constant(anchor_xy_52, dtype=tf.float32)
anchor_xy.append(anchor_xy_52)

anchor_wh = []
anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
anchors_wh_13 = np.array(anchors[6:])
anchors_wh_13 = np.tile(np.reshape(anchors_wh_13, (1, 1, 3, 2)), reps=(13, 13, 1, 1))
anchors_wh_13 = tf.constant(anchors_wh_13, dtype=tf.float32)
anchor_wh.append(anchors_wh_13)

anchors_wh_26 = np.array(anchors[3:6])
anchors_wh_26 = np.tile(np.reshape(anchors_wh_26, (1, 1, 3, 2)), reps=(26, 26, 1, 1))
anchors_wh_26 = tf.constant(anchors_wh_26, dtype=tf.float32)
anchor_wh.append(anchors_wh_26)

anchors_wh_52 = np.array(anchors[:3])
anchors_wh_52 = np.tile(np.reshape(anchors_wh_52, (1, 1, 3, 2)), reps=(52, 52, 1, 1))
anchors_wh_52 = tf.constant(anchors_wh_52, dtype=tf.float32)
anchor_wh.append(anchors_wh_52)

inputs_keras = keras.Input(shape=(416, 416, 3))
yolo4 = yolo_body(inputs=inputs_keras,
                  num_anchors=num_anchors,
                  num_classes=num_classes)

yolo4.load_weights(weight_path)


def save_model(inputs,
               image_shape,
               input_shape,
               num_anchors,
               num_classes,
               score_thres,
               iou_thres,
               scales_xy=None):
    """package model with forward propagation and post processing

    # The output of my model is boxes, scores and classes, so it is
    # very convenient for us to use my saved_model file.
    # You don't need to do postprocessing by yourself.

    Args:
        inputs:      should be fed into the model [keras.Input]
        image_shape: should be fed into the model [keras.Input]
        input_shape: constant
        num_anchors: constant
        num_classes: constant
        score_thres: constant
        iou_thres:   constant
        scales_xy:   eliminate grid sensitivity [tf.constant]

    """

    if scales_xy is None:
        scales_xy = [1.05, 1.1, 1.2]
    scales_xy = tf.constant(scales_xy, dtype=tf.float32)

    input_hw = tf.cast(input_shape[0:2], dtype=tf.float32)
    img_hw = tf.cast(image_shape[0:2], dtype=tf.float32)
    boxes = []
    scores = []

    # [(1, 13, 13, 33), (1, 26, 26, 33), (1, 52, 52, 33)]
    raw_prediction_list = yolo4(inputs)
    for i, raw_prediction in enumerate(raw_prediction_list):

        # bugs happen when i use tf.squeeze in GPU environment
        # bugs: OP_REQUIRES failed at reshape_op.h:57 : Invalid argument: Size 1 must be non-negative, not -9
        # So, avoid using tf.squeeze node when you save model.
        # raw_prediction = tf.squeeze(raw_prediction)
        shape = tf.shape(raw_prediction)
        height = shape[1]
        width = shape[2]

        shape_float = tf.cast(shape[1:3], dtype=tf.float32)

        raw_prediction = keras.backend.reshape(raw_prediction, shape=(height, width, num_anchors, -1))
        box_confidence = tf.sigmoid(raw_prediction[..., 4:5])  # (13, 13, 3, 1)
        box_class_prob = tf.sigmoid(raw_prediction[..., 5:])  # (13, 13, 3, 6)
        box_scores = tf.multiply(box_confidence, box_class_prob)  # (13, 13, 3, 6)
        scores_shape = tf.shape(box_scores)
        num_boxes = scores_shape[0] * scores_shape[1] * scores_shape[2]
        raw_xy = raw_prediction[..., :2]  # (13, 13, 3, 2)
        raw_wh = raw_prediction[..., 2:4]  # (13, 13, 3, 2)

        # 1.decode
        pred_xy = ((scales_xy[i] * tf.sigmoid(raw_xy) - 0.5 * (scales_xy[i] - 1)
                    + anchor_xy[i]) / shape_float[::-1]) * input_hw[::-1]
        pred_wh = (tf.exp(raw_wh) * anchor_wh[i])

        # 2.correct boxes
        scale = tf.reduce_min(input_hw / img_hw)  # tf.float32
        dhw = (input_hw - scale * img_hw) / tf.constant(2.0, dtype=tf.float32)
        correct_yx = ((pred_xy - dhw[::-1]) / scale)[..., ::-1]
        correct_hw = (pred_wh / scale)[..., ::-1]

        # 3.in order to use tf.image.non_max_suppression, the boxes need to be [num_boxes, (y1, x1, y2, x2)]
        correct_box = tf.concat([(correct_yx - correct_hw / 2.0),
                                 (correct_yx + correct_hw / 2.0)], axis=-1)  # (13, 13, 3, 4)
        box_scores = keras.backend.reshape(box_scores, shape=(num_boxes, num_classes))
        correct_box = keras.backend.reshape(correct_box, shape=(num_boxes, 4))
        scores.append(box_scores)
        boxes.append(correct_box)

    # 4.nms for every class
    boxes = tf.concat(boxes, axis=0)
    scores = tf.concat(scores, axis=0)
    boxes_ = []
    scores_ = []
    class_ = []

    # boxes_scores > score_thres
    valid_mask = scores > score_thres
    for i in range(num_classes):
        valid_boxes = tf.boolean_mask(boxes, valid_mask[:, i])
        valid_scores = tf.boolean_mask(scores[:, i], valid_mask[:, i])
        # nms
        index = tf.image.non_max_suppression(boxes=valid_boxes,
                                             scores=valid_scores,
                                             max_output_size=tf.constant(50, dtype=tf.int32),
                                             iou_threshold=iou_thres)
        # gather boxes and scores
        boxes_class = tf.gather(valid_boxes, index)
        scores_class = tf.gather(valid_scores, index)
        # which class of these target boxes
        class_class = tf.ones_like(scores_class, dtype=tf.int32) * i
        boxes_.append(boxes_class)
        scores_.append(scores_class)
        class_.append(class_class)

    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    class_ = tf.concat(class_, axis=0)

    total_model = keras.Model([inputs, image_shape], [boxes_, scores_, class_])
    # 5.save model
    tf.saved_model.save(total_model, save_path)
    return total_model


if __name__ == '__main__':

    inputs_ = keras.Input(shape=(416, 416, 3))
    img_shape = keras.Input(shape=())
    model = save_model(inputs_,
                       img_shape,
                       (416, 416, 3),
                       num_anchors,
                       num_classes,
                       score,
                       iou)

    # test model
    # and through testing, the decoding codes are usable.
    image = Image.open(img_path)
    image_shape = np.array([image.size[1], image.size[0]], dtype='float32')
    image = letterbox_image(image, (416, 416))
    image_array = np.expand_dims(np.array(image, dtype='float32') / 255.0, axis=0)
    boxes, scores, classes = model([image_array, image_shape])
    print(classes)
    print(scores)
    print(boxes)

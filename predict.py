# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:predict.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from nets.yolo4 import yolo_body
import numpy as np

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
                  num_anchors=3,
                  num_classes=6)

yolo4.load_weights('./data/ep290-loss16.428.h5')


def save_model(inputs,
               image_shape,
               input_shape,
               num_anchors,
               num_classes,
               score_thres,
               iou_thres,
               scales_xy=[1.05, 1.1, 1.2]):
    """package model with forward propagation and post processing

    Args:
        inputs:      should be fed into the model
        image_shape: should be fed into the model
        input_shape: constant
        num_anchors: constant
        num_classes: constant
        score_thres: constant
        iou_thres:   constant
        scales_xy:   eliminate grid sensitivity [tf.constant]

    """

    scales_xy = tf.constant(scales_xy, dtype=tf.float32)

    input_hw = tf.cast(input_shape[0:2], dtype=tf.float32)
    img_hw = tf.cast(image_shape[0:2], dtype=tf.float32)
    boxes = []
    scores = []

    # [(1, 13, 13, 33), (1, 26, 26, 33), (1, 52, 52, 33)]
    raw_prediction_list = yolo4.predict(inputs)
    for i, raw_prediction in enumerate(raw_prediction_list):
        raw_prediction = tf.squeeze(raw_prediction)
        shape = tf.shape(raw_prediction)
        height = shape[0]
        width = shape[1]

        shape_float = tf.cast(shape[0:2], dtype=tf.float32)

        raw_prediction = tf.reshape(raw_prediction, shape=(height, width, num_anchors, -1))
        box_confidence = tf.sigmoid(raw_prediction[..., 4:5])  # (13, 13, 3, 1)
        box_class_prob = tf.sigmoid(raw_prediction[..., 5:])  # (13, 13, 3, 6)
        box_scores = tf.multiply(box_confidence, box_class_prob)  # (13, 13, 3, 6)
        raw_xy = raw_prediction[..., :2]  # (13, 13, 3, 2)
        raw_wh = raw_prediction[..., 2:4]  # (13, 13, 3, 2)

        # decode
        pred_xy = ((scales_xy[i] * tf.sigmoid(raw_xy) - 0.5 * (scales_xy[i] - 1)
                    + anchor_xy[i]) / shape_float[::-1]) * input_hw[::-1]
        pred_wh = (tf.exp(raw_wh) * anchor_wh[i])

        # correct boxes
        scale = tf.reduce_min(input_hw / img_hw)  # tf.float32
        dhw = (input_hw - scale * img_hw) / tf.constant(2.0, dtype=tf.float32)
        correct_yx = ((pred_xy - dhw[::-1]) / scale)[..., ::-1]
        correct_hw = (pred_wh / scale)[..., ::-1]
        # in order to use tf.image.non_max_suppression, the boxes need to be [num_boxes, (y1, x1, y2, x2)]
        correct_box = tf.concat([(correct_yx - correct_hw / 2.0),
                                 (correct_yx + correct_hw / 2.0)], axis=-1)  # (13, 13, 3, 4)
        box_scores = tf.reshape(box_scores, shape=(-1, num_classes))
        correct_box = tf.reshape(correct_box, shape=(-1, 4))
        scores.append(box_scores)
        boxes.append(correct_box)

    # nms for every class
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
        # which class
        class_class = tf.ones_like(scores_class, dtype=tf.int32) * i
        boxes_.append(boxes_class)
        scores_.append(scores_class)
        class_.append(class_class)

    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    class_ = tf.concat(class_, axis=0)

    total_model = keras.Model(inputs, [boxes_, scores_, class_])
    # save model
    tf.saved_model.save(total_model, './saved_model')


if __name__ == '__main__':
    inputs = tf.ones(shape=(1, 416, 416, 3))
    img_shape = tf.constant([500, 500, 3])
    save_model(inputs,
               img_shape,
               (416, 416, 3),
               3,
               6,
               0.3,
               0.5)

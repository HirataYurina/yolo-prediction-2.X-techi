# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:visualize.py
# software: PyCharm

from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from nets.utils import letterbox_image
import numpy as np
from argparse import ArgumentParser


def augument_parser():
    parser = ArgumentParser(description='visualize your results to validate your saved_model')
    parser.add_argument('--classes_path', type=str, default='./data/danger_source_classes.txt')
    parser.add_argument('--model_path', type=str, default='./saved_model')
    parser.add_argument('--img_path', type=str, default='./images/danger_source3.jpg')
    args = parser.parse_args()

    return args


arguments = augument_parser()


with open(arguments.classes_path) as f:
    class_names = f.readlines()
    class_names = [name.strip() for name in class_names]


def visualzation(model_path, img_path):
    model = tf.saved_model.load(model_path)
    perdictor = model.signatures['serving_default']
    image = Image.open(img_path)
    image_shape = np.array([image.size[1], image.size[0]], dtype='float32')

    # letter box
    image_letter = letterbox_image(image, (416, 416))
    image_array = np.expand_dims(np.array(image_letter, dtype='float32') / 255.0, axis=0)
    image_constant = tf.constant(image_array, dtype=tf.float32)
    image_shape = tf.constant(image_shape, dtype=tf.float32)

    # infer
    results = perdictor(input_2=image_constant, input_3=image_shape)  # dictionary
    results_ = []
    for key, value in results.items():
        results_.append(value)

    # -------------------------------------------------------------------------------------
    # visualize
    boxes = results_[0]
    scores = results_[1]
    classes = results_[2]

    # starting draw bounding boxes
    font = ImageFont.truetype(font='font/simhei.ttf',
                              size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
    # thickness of bounding box and this thickness is changing according to img_size
    thickness = (image.size[0] + image.size[1]) // 500

    for i, c in list(enumerate(classes)):
        predicted_class = class_names[c]
        box = boxes[i]
        score = scores[i]

        top, left, bottom, right = box
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        # print(label)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for j in range(thickness):
            draw.rectangle([left + j, top + j, right - j, bottom - j],
                           outline=(0, 0, 255))
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                       fill=(128, 128, 128))
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image


if __name__ == '__main__':

    image_show = visualzation(model_path=arguments.model_path,
                              img_path=arguments.img_path)
    image_show.show()
    image_show.save('./images/result2.jpg')

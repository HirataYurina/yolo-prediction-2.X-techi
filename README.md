## Use Tensorflow2.X to Save Model and Predict

*save you model with saved-model format*

*deploy your model with saved-model format*

___

**1. Save Model**

```shell
# save your model that contains postprocessing
# you can directly get boxes, scores and classes by inferrring
python save_model.py -h
```

```shell
usage: save_model.py [-h] [--num_anchors NUM_ANCHORS]
                     [--num_classes NUM_CLASSES] [--score SCORE] [--iou IOU]
                     [--img_path IMG_PATH] [--weight_path WEIGHT_PATH]
                     [--save_path SAVE_PATH]

save your model with saved-model format

optional arguments:
  -h, --help            show this help message and exit
  --num_anchors NUM_ANCHORS
                        number of your anchors
  --num_classes NUM_CLASSES
                        number of your classes
  --score SCORE         score threshold of prediction
  --iou IOU             iou threshold of prediction
  --img_path IMG_PATH   image that you want to predict
  --weight_path WEIGHT_PATH
                        the path of model weights
  --save_path SAVE_PATH
```

**2. Visualize Results**

```shell
# visualize your results and you can validate you saved_model
python visualize.py -h
```

```shell
visualize your results to validate your saved_model

optional arguments:
  -h, --help            show this help message and exit
  --classes_path CLASSES_PATH
  --model_path MODEL_PATH
  --img_path IMG_PATH
```

**3. Results**

<img src="./images/result1.jpg" align="center" width="400px">

<img src="images/result2.jpg" align="center" width="500px">

**4. why i use saved_model format?**

Because saved_model format is a bridge in tensorflow ecology.

If you use **tensorflow serving** to do model persistence and deploy your projects,  saved_model format is a best choice.

If you use **tensorflow lite ** to do mobile projects, saved_model format is a best choice.

<img src="images/tf_ecology.jpg" aligh="center">
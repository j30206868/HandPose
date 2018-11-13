import tensorflow as tf
import numpy as np
import cv2

from model import *

input_tensor = tf.placeholder(tf.float32, shape=(None, None, None,3), name='image')
estimator = HandPose(input_tensor)

TEST_IMAGE_PATH = 'D:/wzchen/PythonProj/tf-openpose/images/hand1.jpg'
image_orig = cv2.cvtColor(cv2.imread(TEST_IMAGE_PATH), cv2.COLOR_BGR2RGB)
estimator.predict(image_orig, 2, True, True)

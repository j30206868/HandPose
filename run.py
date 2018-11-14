import tensorflow as tf
import numpy as np
import cv2

from model import *

input_tensor = tf.placeholder(tf.float32, shape=(None, None, None,3), name='image')
handpose = HandPose(input_tensor)
handpose.load_weights('checkpoints/pretrain')

TEST_IMAGE_PATH = 'hand1.jpg'
image_orig = cv2.imread(TEST_IMAGE_PATH)
handpose.predict(image_orig, 1.5, True, True)

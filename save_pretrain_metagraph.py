import tensorflow as tf
import numpy as np
import cv2

from model import *

input_tensor = tf.placeholder(tf.float32, shape=(None, None, None,3), name='image')
x, net_down_scale = get_tfopenpose_hand_model(input_tensor)
sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())
load_numpy_weights(sess, 'numpy_weights/', "checkpoints/pretrain")
import tensorflow as tf
import numpy as np
import cv2

import tf_pose

from model import *

input_tensor = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='image')
handpose = HandPose(input_tensor)
handpose.load_weights('checkpoints/pretrain')

# TEST_IMAGE_PATH = 'hand1.jpg'
# image_orig = cv2.imread(TEST_IMAGE_PATH)
# handpose.predict(image_orig, 1.5, True, True)



import pdb
import matplotlib.pyplot as plt
import math



TEST_IMAGE_PATH = 'hand2.jpg'
image = cv2.imread(TEST_IMAGE_PATH)
img_w = image.shape[1]
img_h = image.shape[0]
bodys = tf_pose.infer('hand2.jpg')
for body in bodys:
    # l_roi_image = handpose.getLeftHandRoiImage(body.body_parts, image)
    # r_roi_image = handpose.getRightHandRoiImage(body.body_parts, image)
    
#     handpose.fix_predict(l_roi_image)
    lhpts, rhpts = handpose.getHandsPart(image, body)
    
    for partid in lhpts:
        part = lhpts[partid]
        cv2.circle(image, (int(img_w*part.x), int(img_h*part.y)), 10, (255,0,0), 2)
    for partid in rhpts:
        part = rhpts[partid]
        cv2.circle(image, (int(img_w*part.x), int(img_h*part.y)), 10, (255,0,0), 2)
#     for partid in body.body_parts:
#         part = body.body_parts[partid]
#         cv2.circle(image, (int(img_w*part.x), int(img_h*part.y)), 10, (255,0,0), 2)

plt.figure(figsize=(12,12))
plt.imshow(image)
plt.show()
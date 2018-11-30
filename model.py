import tensorflow as tf
import numpy as np
import sys
import os

import cv2
import matplotlib.pyplot as plt

import scipy.stats as st
import math
import pdb

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def make_gauss_var(name, c_i, size=7, sigma=3):
    # with tf.device("/cpu:0"):
    kernel = gauss_kernel(size, sigma, c_i)
    var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
    return var

def get_tfopenpose_hand_model(input_tensor):
    def tf_main_block(input_tensor, padding='SAME'): 
        x = tf.layers.conv2d(input_tensor, 64, 3, strides=(1,1), padding=padding, activation='relu', name='conv1_1')
        x = tf.layers.conv2d(x, 64, 3, strides=(1,1), padding=padding, activation='relu', name='conv1_2')
        x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool1_stage1')

        x = tf.layers.conv2d(x, 128, 3, strides=(1,1), padding=padding, activation='relu', name='conv2_1')
        x = tf.layers.conv2d(x, 128, 3, strides=(1,1), padding=padding, activation='relu', name='conv2_2')
        x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool2_stage1')

        x = tf.layers.conv2d(x, 256, 3, strides=(1,1), padding=padding, activation='relu', name='conv3_1')
        x = tf.layers.conv2d(x, 256, 3, strides=(1,1), padding=padding, activation='relu', name='conv3_2')
        x = tf.layers.conv2d(x, 256, 3, strides=(1,1), padding=padding, activation='relu', name='conv3_3')
        x = tf.layers.conv2d(x, 256, 3, strides=(1,1), padding=padding, activation='relu', name='conv3_4')
        x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='pool3_stage1')

        x = tf.layers.conv2d(x, 512, 3, strides=(1,1), padding=padding, activation='relu', name='conv4_1')
        x = tf.layers.conv2d(x, 512, 3, strides=(1,1), padding=padding, activation='relu', name='conv4_2')
        x = tf.layers.conv2d(x, 512, 3, strides=(1,1), padding=padding, activation='relu', name='conv4_3')
        x = tf.layers.conv2d(x, 512, 3, strides=(1,1), padding=padding, activation='relu', name='conv4_4')
        x = tf.layers.conv2d(x, 512, 3, strides=(1,1), padding=padding, activation='relu', name='conv5_1')
        x = tf.layers.conv2d(x, 512, 3, strides=(1,1), padding=padding, activation='relu', name='conv5_2')
        conv5_3_CPM = tf.layers.conv2d(x, 128, 3, strides=(1,1), padding=padding, activation='relu', name='conv5_3_CPM')
        x = tf.layers.conv2d(conv5_3_CPM, 512, 1, strides=(1,1), padding='VALID', activation='relu', name='conv6_1_CPM')
        conv6_2_CPM = tf.layers.conv2d(x, 22, 1, strides=(1,1), padding='VALID', activation=None, name='conv6_2_CPM')
        return conv5_3_CPM, conv6_2_CPM

    def tf_stage_block(conv5_3_CPM, prev_stage, stage, padding='SAME'):
        x = tf.concat(values=[prev_stage, conv5_3_CPM], axis=3, name='concat_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 128, 7, strides=(1,1), padding=padding, activation='relu', name='Mconv1_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 128, 7, strides=(1,1), padding=padding, activation='relu', name='Mconv2_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 128, 7, strides=(1,1), padding=padding, activation='relu', name='Mconv3_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 128, 7, strides=(1,1), padding=padding, activation='relu', name='Mconv4_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 128, 7, strides=(1,1), padding=padding, activation='relu', name='Mconv5_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 128, 1, strides=(1,1), padding='VALID', activation='relu', name='Mconv6_stage{}'.format(stage))
        x = tf.layers.conv2d(x, 22, 1, strides=(1,1), padding='VALID', activation=None, name='Mconv7_stage{}'.format(stage))
        return x   
    conv5_3_CPM, conv6_2_CPM = tf_main_block(input_tensor)
    prev_stage = conv6_2_CPM
    for stage in range(2, 7):
        prev_stage = tf_stage_block(conv5_3_CPM, prev_stage, stage)
        
    x = prev_stage
    net_down_scale = 8
    return x, net_down_scale

def remove_redundant(parts, heatmap, hand_amount=None):
    ''' remove redundant coordinates for each part
    
    Arguments:

            parts {a list of coordinate list} -- len(parts) == joint_type_amount
                parts = 
                [
                    [[y, x], [y, x], ...], # coordinate list of part 0
                    [[y, x], [y, x], ...], # coordinate list of for part 1
                    [[y, x], [y, x], ...], # coordinate list of for part 2
                                            .
                                            .
                                            .
                    [[y, x], [y, x], ...], # coordinate list of for part n
                ]
                
                once the function executed, 'parts' will be changed
            
            heatmap {heatmap of the image} -- shape: (image_height, image_width, joint_type_amount)
    Returns:
            deleted coordinates for each part
            structure is the same as 'parts' but coordinates are the deleted ones
    '''
    joint_type_amount = len(parts)

    if hand_amount is None:
        hand_amount = int(round(np.average([len(part_id_list) for part_id_list in parts])))
    
    delted_part_list = [[] for i in range(joint_type_amount)]
    for part_id in range(len(parts)):
        joint_coords = parts[part_id]
        jcoord_amount = len(joint_coords)
        if jcoord_amount > hand_amount:
            edges = []
            edge_amount = 0
            for i in range(jcoord_amount):
                for j in range(i+1, jcoord_amount):
                    dist = np.linalg.norm(np.array(joint_coords[i]) - np.array(joint_coords[j]))
                    edges.append([i, j, dist])
                    edge_amount+=1
            # sort by distance in ascending order
            edges.sort(key=lambda x: x[2])
            # delete redundant
            redundant_amount = jcoord_amount - hand_amount
            deleted_amount = 0
            deleted_list = []
            for eid in range(edge_amount):
                if deleted_amount >= redundant_amount:
                    break
                i, j, dist = edges[eid]
                # make sure both i and j is not deleted yet
                if (i in deleted_list) or (j in deleted_list):
                    continue
                i_score = heatmap[joint_coords[i][0], joint_coords[i][1], part_id]
                j_score = heatmap[joint_coords[j][0], joint_coords[j][1], part_id]
                # compare peak score and delete
                if i_score > j_score:
                    deleted_list.append(j)
                else:
                    deleted_list.append(i)
                deleted_amount+=1
            # remove deleted coords from part id list
            # sort id in decending order (so that the deletion of the front element won't affect the rest)
            deleted_list.sort(reverse=True)
            for did in deleted_list:
                delted_part_list[part_id].append(joint_coords[did])
                del parts[part_id][did]
    return delted_part_list

def interchange_yx_to_xy_and_scale(parts, scale_factor):
    for part_id  in range(len(parts)):
        for cid in range(len(parts[part_id])):
            y, x = parts[part_id][cid]
            parts[part_id][cid] = (x*scale_factor, y*scale_factor)

def interchange_yx_to_xy_and_normalize(parts, img_w, img_h):
    for part_id  in range(len(parts)):
        for cid in range(len(parts[part_id])):
            y, x = parts[part_id][cid]
            parts[part_id][cid] = (x/float(img_w), y/float(img_h))

def draw_parts(image, parts, circle_color=(0,127,0), font_color=(0,255,0), font_scale=0.3):
    for part_id  in range(len(parts)):
        for coord in parts[part_id]:
            cv2.circle(image, coord, 10, circle_color)
            cv2.putText(image, str(part_id), coord, cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color)

def fastMax(a, b):
    if (a > b): return a
    else: return b

def getHandROI(target_indice, body_parts, img_w, img_h):
    assert(len(target_indice) == 3)
    if set(target_indice) < set(body_parts):
        ratioWristElbow = 0.33
        shoulder, elbow, wrist = [body_parts[idx] for idx in target_indice]
        distanceWristElbow = math.sqrt(pow((wrist.x - elbow.x)*img_w, 2) + pow((wrist.y - elbow.y)*img_h, 2))
        distanceElbowShoulder = math.sqrt(pow((shoulder.x - elbow.x)*img_w, 2) + pow((shoulder.y - elbow.y)*img_h, 2))
        box_w = 1.5 * fastMax(distanceWristElbow, 0.9 * distanceElbowShoulder)
        x = (wrist.x + ratioWristElbow * (wrist.x - elbow.x)) * img_w
        y = (wrist.y + ratioWristElbow * (wrist.y - elbow.y)) * img_h
        x1 = x - box_w / 2.0
        y1 = y - box_w / 2.0
        x2 = x1 + box_w
        y2 = y1 + box_w
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img_w-1)
        y2 = min(y2, img_h-1)
        return [int(round(x1)), int(round(y1)), int(round(x2))+1, int(round(y2))+1]
        # return [int(round(x1)), int(round(y1)), int(round(x2-x1+1)), int(round(y2-y1+1))]
    else:
        return None

class HandPart:
    """
    size: 'l' or 'r'
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('side', 'part_idx', 'x', 'y', 'score')

    def __init__(self, side, part_idx, x, y, score):
        self.side = side
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return '%s Part %d' % (self.side, self.part_idx)

    def __str__(self):
        return 'HandPart:%s %d-(%.2f, %.2f) score=%.2f' % (self.side, self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()

class HandPose:
    # l_shoulder_id = 5
    # l_elbow_id = 6
    # l_wrist_id = 7
    # r_shoulder_id = 2
    # r_elbow_id = 3
    # r_wrist_id = 4
    # left_indice = [l_shoulder_id, l_elbow_id, l_wrist_id]
    # right_indice = [r_shoulder_id, r_elbow_id, r_wrist_id]

    def __init__(self, input_tensor):
        self.coco_left_arm_body_indice = [5, 6, 7]
        self.coco_right_arm_body_indice = [2, 3, 4]
        self.input_tensor = input_tensor
        self.sess = tf.Session(graph=tf.get_default_graph())
        ###
        self.openpose, self.netscale = get_tfopenpose_hand_model(input_tensor)
        self.openpose_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ###
        self.is_pred_init = False
        ###
        self.joint_amount = self.openpose.shape[-1].value - 1

    def load_weights(self, tfckpt_path):
        saver = tf.train.Saver(var_list=self.openpose_var_list)
        saver.restore(self.sess, tfckpt_path)
        print('OpenPose pretrain weights are LOADED!')
        
    def init_predict_model(self):
        with tf.variable_scope('CWZHandPosePredict'):
            gaussian_heatMat = tf.nn.depthwise_conv2d(
                self.openpose, 
                make_gauss_var('gaussian_filter', self.openpose.get_shape()[3].value), 
                [1, 1, 1, 1], padding='SAME'
            )
            max_pooled_in_tensor = tf.nn.pool(
                gaussian_heatMat, 
                window_shape=(3, 3), 
                pooling_type='MAX', 
                padding='SAME'
            )
            self.peaks_tensor = tf.where(
                tf.equal(gaussian_heatMat, max_pooled_in_tensor), 
                gaussian_heatMat,
                tf.zeros_like(gaussian_heatMat)
            )
        # self.sess.run(tf.global_variables_initializer())
        ### initialize tensor used created by prediction tensorflow ops
        uninit_vars = [var for var in tf.global_variables() if 'CWZHandPosePredict' in var.name]
        self.sess.run(tf.variables_initializer(uninit_vars))
        ###
        self.is_pred_init = True

    def getLeftHandRect(self, body_parts, img_w, img_h):
        return getHandROI(self.coco_left_arm_body_indice, body_parts, img_w, img_h)

    def getRightHandRect(self, body_parts, img_w, img_h):
        return getHandROI(self.coco_right_arm_body_indice, body_parts, img_w, img_h)

    def getLeftHandRoiImage(self, body_parts, image):
        lbox = getHandROI(self.coco_left_arm_body_indice, body_parts, image.shape[1], image.shape[0])
        if lbox is not None:
            return image[lbox[1]:lbox[3],lbox[0]:lbox[2],:], lbox
        return None, None

    def getRightHandRoiImage(self, body_parts, image):
        rbox = getHandROI(self.coco_right_arm_body_indice, body_parts, image.shape[1], image.shape[0])
        if rbox is not None:
            return image[rbox[1]:rbox[3],rbox[0]:rbox[2],:], rbox
        return None, None

    # for debug
    def predict(self, image_orig, scale_mul = 1.5, show_pred=False, show_removed=False):
        if not self.is_pred_init:
            self.init_predict_model()
    
        scale = 368/image_orig.shape[1]
        scale = scale*scale_mul
        ###
        image = cv2.resize(image_orig, (0,0), fx=scale, fy=scale) 
        # net_out.shape = input_tensor.shape / self.netscale
        net_out = self.sess.run(self.peaks_tensor, feed_dict={self.input_tensor: np.expand_dims( image /256 -0.5 ,0)})[0]

        peaks = np.argwhere(net_out >= 0.1)
        parts = [[] for i in range(22)]
        for y, x, part_id in peaks:
            # y = peak[0], x = peak[1], part_id = peak[2]
            parts[part_id].append(np.array([y, x]))

        deleted_parts = remove_redundant(parts, net_out)  
        interchange_yx_to_xy_and_scale(parts, self.netscale)  
        
        if show_pred:
            draw_parts(image, parts, (0,127,0), (0,255,0))

            if show_removed:
                interchange_yx_to_xy_and_scale(deleted_parts, self.netscale)  
                draw_parts(image, deleted_parts, (0,0,127), (0,0,255))
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12,12))
            plt.imshow(image)
            plt.show()

        return parts

    def fix_predict(self, image, handside, resize=True, hand_amount=None):
        if not self.is_pred_init:
            self.init_predict_model()
    
        if resize:
            image = cv2.resize(image, (368, 368))
        net_out = self.sess.run(self.peaks_tensor, feed_dict={self.input_tensor: np.expand_dims( image /256 -0.5 ,0)})[0]

        peaks = np.argwhere(net_out >= 0.1)
        parts = [[] for i in range(22)]
        for y, x, part_id in peaks:
            # y = peak[0], x = peak[1], part_id = peak[2]
            parts[part_id].append(np.array([y, x]))

        remove_redundant(parts, net_out, hand_amount=hand_amount)  
        interchange_yx_to_xy_and_normalize(parts, image.shape[1]/self.netscale, image.shape[2]/self.netscale)  

        hand = {}
        for jid in range(self.joint_amount):
            if len(parts[jid]) > 0:
                x = parts[jid][0][0]
                y = parts[jid][0][1]
                hand[jid] = HandPart(handside, jid, x, y, 1)

        return hand

    def getHandsPart(self, image, body):
        l_roi_img, lbox = self.getLeftHandRoiImage(body.body_parts, image)
        r_roi_img, rbox = self.getRightHandRoiImage(body.body_parts, image)
        
        left_hand_parts = {}
        right_hand_parts = {}
        if l_roi_img is not None:
            left_hand_parts = self.fix_predict(l_roi_img, 'l', hand_amount=1)

        if r_roi_img is not None:
            right_hand_parts = self.fix_predict(r_roi_img, 'r', hand_amount=1)
        
        lbox_w = lbox[2] - lbox[0]
        lbox_h = lbox[3] - lbox[1]
        for id in left_hand_parts:
            left_hand_parts[id].x = lbox[0] + left_hand_parts[id].x*lbox_w
            left_hand_parts[id].y = lbox[1] + left_hand_parts[id].y*lbox_h

        rbox_w = rbox[2] - rbox[0]
        rbox_h = rbox[3] - rbox[1]
        for id in right_hand_parts:
            right_hand_parts[id].x = rbox[0] + right_hand_parts[id].x*rbox_w
            right_hand_parts[id].y = rbox[1] + right_hand_parts[id].y*rbox_h

        return (left_hand_parts, right_hand_parts)
import tensorflow as tf
import numpy as np
import sys
import os

import cv2
import matplotlib.pyplot as plt

import scipy.stats as st
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

def remove_redundant(parts, heatmap):
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

def draw_parts(image, parts, circle_color=(0,127,0), font_color=(0,255,0), font_scale=0.3):
    for part_id  in range(len(parts)):
        for coord in parts[part_id]:
            cv2.circle(image, coord, 10, circle_color)
            cv2.putText(image, str(part_id), coord, cv2.FONT_HERSHEY_COMPLEX, font_scale, font_color)


def load_numpy_weights(sess, numpy_weights_dir, save_path):
    if (os.path.isdir(numpy_weights_dir)):
        print('numpy_weights_dir %s doesn\'t exist' % numpy_weights_dir)

    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        layer_name = variable.name[:variable.name.find('/')]
        if (os.path.exists(os.path.join(numpy_weights_dir, "W_%s.npy" % layer_name))):
            print('Loading %s' % variable)
            if ('kernel' in variable.name):
                w = np.array(np.load(os.path.join(numpy_weights_dir, "W_%s.npy" % layer_name)).tolist())
                sess.run(variable.assign(w))
            elif ('bias' in variable.name):
                b = np.array(np.load(os.path.join(numpy_weights_dir, "b_%s.npy" % layer_name)).tolist())
                sess.run(variable.assign(b))
            else:
                print("Load variable %s failed, exit\n" % variable.name);
                sys.exit()
        else:
            print("Variable %s weight file not found, exit\n" % variable.name);
            sys.exit()
    # save pretrained weight
    saver = tf.train.Saver()
    saver.save(sess, save_path)

class HandPose:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.sess = tf.Session(graph=tf.get_default_graph())
        ###
        self.openpose, self.netscale = get_tfopenpose_hand_model(input_tensor)
        self.openpose_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        saver = tf.train.Saver(var_list=self.openpose_var_list)
        saver.restore(self.sess, "checkpoints/pretrain")
        print('OpenPose pretrain weights are LOADED!')
        ###
        self.is_pred_init = False
        
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
        
    def predict(self, image_orig, scale_mul = 1.5, show_pred=False, show_removed=False):
        if not self.is_pred_init:
            self.init_predict_model()
    
        scale = 368/image_orig.shape[1]
        scale = scale*scale_mul
        ###
        image = cv2.resize(image_orig, (0,0), fx=scale, fy=scale) 
        net_out = self.sess.run(self.peaks_tensor, feed_dict={self.input_tensor: np.expand_dims( image /256 -0.5 ,0)})[0]

        peaks = np.argwhere(net_out > 0.1)
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
                
            plt.figure(figsize=(12,12))
            plt.imshow(image)
            plt.show()

        return parts
# input_tensor = tf.placeholder(tf.float32, shape=(None, None, None,3), name='image')
# x, net_down_scale = get_tfopenpose_hand_model(input_tensor)
# sess = tf.Session(graph=tf.get_default_graph())
# sess.run(tf.global_variables_initializer())
# load_numpy_weights(sess, "./openpose_handmodel_numpy_weight", "./openpose_hand_pretrain")
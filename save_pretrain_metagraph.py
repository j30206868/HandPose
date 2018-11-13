import tensorflow as tf
import numpy as np
import cv2

from model import get_tfopenpose_hand_model

def load_numpy_weights(sess, numpy_weights_dir, var_list=None):
    '''Load numpy weights into tensorflow variable

        This function can only load kernel and bias weights
        Numpy weight file should be named as follow:
            Layer Name: 'conv1_1'
            Kernel file: ${numpy_weights_dir}/W_conv1_1
            Bias file: ${numpy_weights_dir}/b_conv1_1

        Kernel weights should be kept in the same format as the built 
        Tensorflow Model.
        All the weights will be directly loaded without transpose. 
            (Tensorflow default, N=1 'NWC', N=2 'NHWC', N=3 'NDHWC')

    Arguments:
        sess {tf.Session} -- [session]
        numpy_weights_dir {string} -- 
            [a directory path which keeps all the numpy weight files]
        var_list {list} --
            [a list of target varaibles' tensor or name]
    '''

    if (os.path.isdir(numpy_weights_dir)):
        print('numpy_weights_dir %s doesn\'t exist' % numpy_weights_dir)

    if var_list is None:
        # get default var_list
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    for variable in var_list:
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
    

input_tensor = tf.placeholder(tf.float32, shape=(None, None, None,3), name='image')
# Build OpenPose Hand Detection model in Tensorflow
x, net_down_scale = get_tfopenpose_hand_model(input_tensor)
sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())
# Load numpy weights to variables according to varaible name
load_numpy_weights(sess, 'numpy_weights/', var_list=None)
# save to meta graph
saver = tf.train.Saver()
saver.save(sess, "checkpoints/pretrain")
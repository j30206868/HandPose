import tensorflow as tf
import numpy as np
import os

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
        print('Loading %s' % variable)
        path = ''
        if ('kernel' in variable.name):
            path = os.path.join(numpy_weights_dir, "W_%s.npy" % layer_name)
        elif ('bias' in variable.name):
            path = os.path.join(numpy_weights_dir, "b_%s.npy" % layer_name)
        else:
            print("Variable %s is not a kernel or bias weights, loading failed and exit\n" % variable.name)
            sys.exit()

        print('\tfrom %s' % path)
        if os.path.exists(path):
            w = np.array(np.load(path).tolist())
            print('\tnumpy weight array shape: ', w.shape)
            sess.run(variable.assign(w))
        else:
            print("Variable %s weight file not found, exit\n" % variable.name);
            sys.exit()
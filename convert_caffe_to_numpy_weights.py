import numpy as np
import os
import sys

CAFFE_PYTHON_PATH='D:/wzchen/caffe/build/install/python'
caffe_weights_dir = 'numpy_weights/'
caffe_model = './pose_iter_102000.caffemodel'
caffe_proto = './pose_deploy.prototxt'

def save_caffe_numpy_weights(caffe_weights_dir, convert_to_tf_order=True):
    print('\nCaffe Model Summary:')
    for layer_name, blob in net.blobs.items():
        print(' ',layer_name, blob.data.shape)

    print('\nStart save caffe model into numpy weights...')
    # write out weight matrices and bias vectors
    for layer_name, varaibles in net.params.items():
        kernel = varaibles[0].data
        bias   = varaibles[1].data
        print(layer_name, kernel.shape, bias.shape)
        if convert_to_tf_order:
            kernel = np.transpose(kernel, (2, 3, 1, 0))
        np.save(os.path.join(caffe_weights_dir, "W_%s.npy" % layer_name), kernel)
        np.save(os.path.join(caffe_weights_dir, "b_%s.npy" % layer_name), bias)

    print("save_caffe_numpy_weights() finished")

if (not os.path.isdir(caffe_weights_dir)):
    os.mkdir(caffe_weights_dir)

sys.path.append(CAFFE_PYTHON_PATH)
import caffe
caffe.set_mode_cpu()
net = caffe.Net(caffe_proto, caffe_model, caffe.TEST)
save_caffe_numpy_weights(caffe_weights_dir, convert_to_tf_order=True)
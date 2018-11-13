import tensorflow as tf
from tf_numpy_weights_loader import load_numpy_weights
from model import get_tfopenpose_hand_model

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
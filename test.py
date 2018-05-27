import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x = tf.constant(1.0)
with tf.Session():
    print(x.eval())

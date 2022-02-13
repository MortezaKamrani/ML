
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers.normalization import batch_normalization
np.random.seed(1)
n = np.random.randn(3,3,2,4)

f = np.argmax(n, axis=-1)
c = tf.math.reduce_max(n, axis=-1)

e = (c >= 1.3)
tf.image.no
f = tf.boolean_mask(f, e)
c = tf.boolean_mask(c, e)
n = tf.boolean_mask(n, e)

print(e)
print(f)
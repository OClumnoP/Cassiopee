
import tensorflow as tf

dataset = tf.data.Dataset.range(3)

dataset = dataset.prefetch(2)

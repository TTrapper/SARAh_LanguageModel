import tensorflow as tf
import numpy as np
import time 
import layers

seq_lens = np.random.randint(32, size=64)

tf.reset_default_graph()
with tf.Session() as sess:

    values = tf.constant(np.random.rand(64, 32, 2048))
    keys = tf.constant(np.random.rand(64, 32, 1024))
    query = tf.constant(np.random.rand(64, 1024))
    num_heads = 8
    attended = layers.multihead_attention(values, keys, query, seq_lens, num_heads)
    sess.run(tf.global_variables_initializer())
    print 'running multihead_attention'
    start = time.time()
    for i in range(100):
        sess.run(attended)
    print 'time:',  time.time()-start


tf.reset_default_graph()
with tf.Session() as sess:
    values = tf.constant(np.random.rand(64, 32, 2048))
    keys = tf.constant(np.random.rand(64, 32, 1024))
    query = tf.constant(np.random.rand(64, 1024))
    num_heads = 8
    attended = layers.slow_multihead_attention(values, keys, query, seq_lens, num_heads)
    sess.run(tf.global_variables_initializer())
    print 'running slow_multihead_attention'
    start = time.time()
    for i in range(100):
         sess.run(attended)
    print 'time:', time.time()-start

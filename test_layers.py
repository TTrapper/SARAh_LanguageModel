import unittest

import tensorflow as tf
import numpy as np

import layers

def initialize_vars(sess):
    sess.run(tf.global_variables_initializer())

class TestFeedForward(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
            outputs = layers.feed_forward(inputs, num_nodes=20)
            initialize_vars(sess)
            outputs = sess.run(outputs)
            self.assertEqual(outputs.shape, (3, 20))

    def test_dropout(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = tf.constant([[0.5, 0.5], [1, 1], [2, 2]], dtype=tf.float32)
            with tf.variable_scope('dropout_test'):
                outputs_dropped = layers.feed_forward(inputs, num_nodes=1024, keep_prob=0.5)
            with tf.variable_scope('dropout_test', reuse=True):
                outputs_kept = layers.feed_forward(inputs, num_nodes=1024, keep_prob=1.0)
            sess.run(tf.global_variables_initializer())
            outputs_dropped = sess.run(outputs_dropped)
            outputs_kept = sess.run(outputs_kept)
            # values that haven't been dropped should be the same (after scaling),
            # therefore the only values that differ should be the dropped ones.
            zero_indices = np.where(np.not_equal(outputs_dropped, 2*outputs_kept))
            self.assertEqual(np.sum(outputs_dropped[zero_indices]), 0)
            # roughly half of the values should be dropped
            drop_rate = float(zero_indices[0].size)/outputs_kept.size
            self.assertTrue(abs(0.5 - drop_rate) < 0.05)

    def test_layer_norm(self):
        tf.reset_default_graph()
        def check_for_var(varname, count): 
            var = filter(lambda var: 'gamma' in var.name, tf.trainable_variables())
            self.assertEqual(len(var), count)
        with tf.Session() as sess:
            inputs = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
            outputs = layers.feed_forward(inputs, num_nodes=20, layer_norm=False)
            check_for_var('gamma', 0)
            check_for_var('beta', 0)
            with tf.variable_scope('norm'):
                outputs = layers.feed_forward(outputs, num_nodes=10, layer_norm=True)
                check_for_var('gamma', 1)
                check_for_var('beta', 1)

    def test_activation_fn(self):
        tf.reset_default_graph()
        def to_zero(tensor):
            return tensor*0
        with tf.Session() as sess:
            inputs = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
            outputs = layers.feed_forward(inputs, num_nodes=20, activation_fn=to_zero)
            initialize_vars(sess)
            outputs = sess.run(outputs)
            self.assertEqual(np.sum(outputs), 0.0)

class Test_MLP(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = tf.constant([[0, 0], [1, 1], [2, 2]], dtype=tf.float32)
            layer_specs = [{'num_nodes':20}, {'num_nodes':30}]
            outputs = layers.mlp(inputs, layer_specs)
            initialize_vars(sess)
            sess.run(outputs)
            self.assertEqual(outputs.shape, (3, 30))

class Test_Attention(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            values = tf.constant(np.random.rand(3, 6, 24))
            keys = tf.constant(np.random.rand(3, 6, 12))
            query = tf.constant(np.random.rand(3, 12))
            seq_lens = [1, 4, 5]
            num_heads = 2
            attended = layers.multihead_attention(values, keys, query, seq_lens, num_heads)
            sess.run(tf.global_variables_initializer())
            attended = sess.run(attended)
            self.assertEqual(attended.shape, (3, 24))

    def test_multi_head(self):
        tf.reset_default_graph()
        # compare slow_multihead_attention to multihead_attention.
        # the result should be the same.
        with tf.Session() as sess: 
            values = tf.constant(np.random.rand(3, 6, 64))
            keys = tf.constant(np.random.rand(3, 6, 64))
            query = tf.constant(np.random.rand(3, 64))
            seq_lens = [2, 3, 6]
            num_heads = 8
            attended = layers.multihead_attention(values, keys, query, seq_lens, num_heads)
            attended_slow = layers.slow_multihead_attention(values, keys, query, seq_lens, num_heads)
            sess.run(tf.global_variables_initializer())
            attended = sess.run(attended)
            attended_slow = sess.run(attended_slow)
            diff = attended - attended_slow
            self.assertTrue(np.greater(1e-15, diff).all())

    def test_seq_masking(self):
        tf.reset_default_graph() 
        with tf.Session() as sess: 
            values = tf.constant(np.random.rand(3, 6, 4))
            # make keys identitcal so that attention will just average elements
            keys = tf.constant(np.ones([3, 6, 4]))
            query = tf.constant(np.random.rand(3, 4))
            seq_lens = [1, 3, 6]
            num_heads = 1
            attended = layers.multihead_attention(values, keys, query, seq_lens, num_heads)
            sess.run(tf.global_variables_initializer())
            attended = sess.run(attended)
            values = sess.run(values)
            # 1st batch element has sequence len 1, attention should just select it
            self.assertTrue(np.array_equal(attended[0], values[0, 0, :]))
            # 2nd batch element has sequence len 3, attention should average them
            diff = attended[1] - np.mean(values[1, :3, :], axis=0)
            self.assertTrue(np.greater(1e-15, diff).all())
            # 3rd batch element has sequence len 6, attention should average them
            diff = attended[2] - np.mean(values[2, :, :], axis=0)
            self.assertTrue(np.greater(1e-15, diff).all())


class Test_SARAh(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            with tf.variable_scope('SARAh', dtype=tf.float16):
                inputs = tf.constant(np.random.rand(3, 6, 128), dtype=tf.float16)
                seq_lens = [6, 3, 4]
                outputs = layers.sarah(inputs, seq_lens, val_size=96, key_size=32, num_heads=4)
                sess.run(tf.global_variables_initializer())
                outputs = sess.run(outputs)
                self.assertEqual(outputs.shape, (3, 6, 128))
            with tf.variable_scope('SARAh_external_mem', dtype=tf.float16):
                inputs = tf.constant(np.random.rand(3, 6, 160), dtype=tf.float16)
                external_mem = tf.constant(np.random.rand(3, 13, 128), dtype=tf.float16)
                outputs = layers.sarah(inputs, seq_lens, val_size=96, key_size=32, num_heads=4,
                    external_mem_3=external_mem, external_seq_lens_1=[2, 4, 13])
                sess.run(tf.global_variables_initializer())
                outputs = sess.run(outputs)
                self.assertEqual(outputs.shape, (3, 6, 128+32))

    def test_multilayer_compiles(self):
        with tf.Session() as sess:
            with tf.variable_scope('sarah_multilayer', dtype=tf.float16):
                inputs = tf.constant(np.random.rand(3, 6, 128), dtype=tf.float16)
                seq_lens = [6, 3, 4]
                layer_specs = [{'val_size': 96,
                           'key_size': 32,
                           'num_heads': 4,
                           'keep_prob': 0.5,
                           'activation_fn': None},
                          {'val_size': 128,
                           'key_size': 64,
                           'num_heads': 2,
                           'keep_prob': 1.0,
                           'activation_fn': tf.nn.relu}]
                outputs, out_by_layer = layers.sarah_multilayer(inputs, seq_lens, layer_specs)
                sess.run(tf.global_variables_initializer())
                outputs = sess.run(outputs)
                self.assertEqual(outputs.shape, (3, 6, 128 + 64))

            with tf.variable_scope('sarah_multilayer_external_mem', dtype=tf.float16):
                inputs = tf.constant(np.random.rand(3, 12, 128), dtype=tf.float16)
                external_seq_lens = seq_lens # outputs above will be conditionin seq below
                seq_lens = [6, 3, 12]
                layer_specs = [{'val_size': 96,
                           'key_size': 32,
                           'num_heads': 4,
                           'keep_prob': 0.5,
                           'external_mem_3': out_by_layer[0],
                           'external_seq_lens_1': external_seq_lens,
                           'activation_fn': None},
                          {'val_size': 128,
                           'key_size': 64,
                           'num_heads': 2,
                           'keep_prob': 1.0,
                           'external_mem_3': out_by_layer[1],
                           'external_seq_lens_1': external_seq_lens,
                           'activation_fn': tf.nn.relu}]
                outputs, _ = layers.sarah_multilayer(inputs, seq_lens, layer_specs)
                sess.run(tf.global_variables_initializer())
                outputs = sess.run(outputs)
                self.assertEqual(outputs.shape, (3, 12, 256))

    def test_bidirectional_multilayer_compiles(self):
        with tf.Session() as sess:
            with tf.variable_scope('sarah_bidirectional_multilayer', dtype=tf.float16):
                inputs = tf.constant(np.random.rand(3, 6, 128), dtype=tf.float16)
                seq_lens = [6, 3, 4]
                layer_specs = 2*[{
                           'val_size': 96,
                           'key_size': 32,
                           'num_heads': 4,
                           'keep_prob': 0.5,
                           'activation_fn': None,
                           'bidirectional': True}]
                outputs, out_by_layer = layers.sarah_multilayer(inputs, seq_lens, layer_specs)
                sess.run(tf.global_variables_initializer())
                outputs = sess.run(outputs)
                self.assertEqual(outputs.shape, (3, 6, 128))

            with tf.variable_scope('sarah_bidir_external_mem', dtype=tf.float16):
                inputs = tf.constant(np.random.rand(3, 12, 128), dtype=tf.float16)
                external_seq_lens = seq_lens # outputs above will be conditionin seq below
                seq_lens = [6, 3, 12]
                layer_specs = [{
                           'val_size': 96,
                           'key_size': 32,
                           'num_heads': 4,
                           'keep_prob': 0.5,
                           'external_mem_3': out_by_layer[1],
                           'external_seq_lens_1': external_seq_lens,
                           'activation_fn': None}]
                outputs, _ = layers.sarah_multilayer(inputs, seq_lens, layer_specs)
                sess.run(tf.global_variables_initializer())
                outputs = sess.run(outputs)
                self.assertEqual(outputs.shape, (3, 12, 160))
if __name__ == '__main__':
    unittest.main()

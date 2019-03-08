import os
import unittest

import tensorflow as tf
import numpy as np

import config
import data_pipe
import layers
import model_def
import train

class TestModel(unittest.TestCase):
    def test_compiles(self):

        tf.reset_default_graph()
        with tf.Session() as sess:
            conf = config.get_config(keep_prob=1.0)
            data = data_pipe.Data('./example_data/processed/', conf['batch_size'])
            sos_token = 0
            model = model_def.Model(data.src, data.trg, len(data.id_to_chr), sos_token, conf)
            data.initialize(sess, data.datadir + '*')
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            out_logits_3 = sess.run(model.out_logits_3)

    def test_batch_size(self):
        # Test that batch_size doesn't affect output
        # FIXME small deviance in values, TOLERANCE should be exactly 0
        TOLERANCE = 1e-2
        tf.reset_default_graph()
        with tf.Session() as sess:
            conf = config.get_config(keep_prob=1.0)
            sos_token = 0
            conf['batch_size'] = 8
            data_batched = data_pipe.Data('./example_data/processed/', conf['batch_size'], shuffle_buffer=1)
            model = model_def.Model(data_batched.src, data_batched.trg, len(data_batched.id_to_chr), sos_token, conf)
            out_batched = model.out_logits_3
            tf.get_variable_scope().reuse_variables()
            conf['batch_size'] = 1
            data_single = data_pipe.Data('./example_data/processed/', conf['batch_size'], shuffle_buffer=1)
            model = model_def.Model(data_single.src, data_single.trg, len(data_single.id_to_chr), sos_token, conf)
            out_single = model.out_logits_3

            data_batched.initialize(sess, data_batched.datadir + '*')
            data_single.initialize(sess, data_single.datadir + '*')
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())

            out_batched, out_single = sess.run([out_batched, out_single])
            print "ABSOLUTE DIFF BETWEEN BATCHED OUTPUTS AND SINGLE OUTPUTS"
            print abs(out_batched[0] - out_single[0])
            self.assertTrue((abs(out_batched[0] - out_single[0]) <= TOLERANCE).all())


    def test_inference_vs_train(self):
        # Check that model outputs are identical when running inference mode
        # FIXME small deviance in values, TOLERANCE should be exactly 0
        TOLERANCE = 1e-3
        tf.reset_default_graph()
        with tf.Session() as sess:
            conf = config.get_config(keep_prob=1.0)
            conf['batch_size'] = 1
            data = data_pipe.Data('./example_data/processed/', conf['batch_size'])
            model, free_model = train.build_model(data, conf)
            data.initialize(sess, data.datadir + '*')
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            (out_logits_3, src, trg) = sess.run([model.out_logits_3, data.src, data.trg])
            src = data.array_to_strings(src)[0]
            trg = data.array_to_strings(trg)[0]
            feed = {data.src_place:src, data.trg_place:trg}
            free_logits_3 = sess.run(free_model.out_logits_3, feed_dict=feed)
            # Check that the model's outputs are the same regardless of what data pipeline is used
            self.assertTrue((np.abs(out_logits_3[0] - free_logits_3) <= TOLERANCE).all())

            # Run the inference model as though generating one char at time, and check the outputs
            feed = {data.src_place:src, data.trg_place:''.ljust(conf['chrs_per_word'])}
            free_logits_3 = sess.run(free_model.out_logits_3, feed_dict=feed)
            self.assertTrue((np.abs(free_logits_3[0,0,:] - out_logits_3[0,0,:]) <= TOLERANCE).all())
            print src
            print trg
            print "RUNNING INFERENCE (slowly)"
            trg_so_far = ''
            for index, char in enumerate(trg):
                trg_so_far += char
                # Pad the trg so the chrs can be evenly divided
                padlen = conf['chrs_per_word'] - (len(trg_so_far) % conf['chrs_per_word'])
                padstr = ''.ljust(padlen)
                feed = {data.src_place:src, data.trg_place:trg_so_far + padstr}
                free_logits_3 = sess.run(free_model.out_logits_3, feed_dict=feed)
                self.assertTrue((abs(free_logits_3[0, index, :] - out_logits_3[0, index, :]) <= TOLERANCE).all())

if __name__ == '__main__':
    unittest.main()

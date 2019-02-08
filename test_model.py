import os
import unittest

import tensorflow as tf
import numpy as np

import data_pipe
import model_def


class TestModel(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 20
            max_line_len = 64
            basedir = './example_data/train_and_eval/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)

            word_encoder_layers = 4*[{'num_nodes':64,
                                      'activation_fn':tf.nn.relu,
                                      'layer_norm':True,
                                      'keep_prob':0.9}]

            sentence_encoder_layers = 2*[{'val_size':64,
                                         'key_size':32,
                                         'num_heads':2,
                                         'keep_prob':0.9,
                                         'activation_fn':tf.nn.relu,
                                         'bidirectional':True}]

            sentence_decoder_layers = 2*[{'val_size':64,
                                         'key_size':32,
                                         'num_heads':2,
                                         'keep_prob':0.9,
                                         'activation_fn':tf.nn.relu,
                                         'bidirectional':False}]

            word_decoder_sarah = {'val_size':64,
                                  'key_size':32,
                                  'num_heads':2,
                                  'keep_prob':0.9,
                                  'activation_fn':tf.nn.relu,
                                  'bidirectional':False}

            word_decoder_mlp = 4*[{'num_nodes':64,
                                   'activation_fn':tf.nn.relu,
                                   'layer_norm':True,
                                   'keep_prob':0.9}]

            config = {'char_embed_size':32,
                      'spell_vector_len':20,
                      'keep_prob':0.9,
                      'word_encoder_layers':word_encoder_layers,
                      'sentence_encoder_layers':sentence_encoder_layers,
                      'sentence_decoder_layers':sentence_decoder_layers,
                      'word_decoder_sarah':word_decoder_sarah,
                      'word_decoder_mlp':word_decoder_mlp}

            with tf.variable_scope('Model', dtype=tf.float16):
                model = model_def.Model(data.src,
                                        data.src_sentence_len,
                                        data.trg_word_enc,
                                        data.trg_sentence_len,
                                        data.trg_word_dec,
                                        data.trg_word_len,
                                        len(data.chr_to_freq),
                                        config)

            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            data.initialize(sess, data.traindir + '*')
            sess.run(model.out_logits_4)


if __name__ == '__main__':
    unittest.main()

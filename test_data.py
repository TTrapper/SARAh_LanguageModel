import os
import unittest

import tensorflow as tf
import numpy as np

import data_pipe

class TestPipeline(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 20
            max_line_len = 64
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            src_op, trg_op = iterator.get_next()
            src, trg = sess.run([src_op, trg_op])

    def test_produces_expected(self):
        src0 = "ALBERT EINSTEIN REFERENCE ARCHIVE             \nData should be ascii or otherwise encode " +\
               "1 byte per character: Tensorflow's string_split doesn't play nice otherwise."
        trg0 = "___ _RELATIVITY:_ _THE_ _SPECIAL_ _AND_ _GENERAL_ _THEORY_ ___      \n___ _Words_ " +\
               "_are_ _assumed_ _to_ _be_ _be_ _delimited_ _by_ _a_ _single_ _space_ _character._ ___"
        src1 = "RELATIVITY: THE SPECIAL AND GENERAL THEORY      \nWords are assumed to be be delimited" +\
               " by a single space character."
        trg1 = "___ _BY_ _ALBERT_ _EINSTEIN_ ___                 \n___ _The_ _source/target_ _pairs_ _are_ " +\
               "_selected_ _by_ _choosing_ _concurrent_ _lines,_ _so_ _each_ _line_ _is_ " +\
               "_considered_ _the_ _target_ _for_ _the_ _previous_ _line._ ___"

        def next_batch(sess, src_op, trg_op):
            src, trg = sess.run([src_op, trg_op])
            src = data_pipe.char_array_to_txt(src)
            src = data_pipe.replace_pad_chrs(src)
            trg = data_pipe.char_array_to_txt(trg)
            trg = data_pipe.replace_pad_chrs(trg)
            return src, trg

        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 1024
            max_line_len = 1024
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                cycle_length=2, shuffle_buffer=1)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            src_op, trg_op = [tf.sparse_tensor_to_dense(t, default_value=chr(1))
                for t in iterator.get_next()]
            src, trg = next_batch(sess, src_op, trg_op)
            self.assertEqual(src, src0)
            self.assertEqual(trg, trg0)
            src, trg = next_batch(sess, src_op, trg_op)
            self.assertEqual(src, src1)
            self.assertEqual(trg, trg1)

    def test_max_word_len(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 4
            max_line_len = 1024
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            src_op, trg_op = iterator.get_next()
            src, trg = sess.run([src_op, trg_op]) 
            self.assertEqual(src.dense_shape[2], max_word_len)
            self.assertEqual(trg.dense_shape[2], max_word_len + 2)

    def test_max_line_len(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 1024
            max_line_len = 4
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            src_op, trg_op = iterator.get_next()
            src, trg = sess.run([src_op, trg_op]) 
            self.assertEqual(src.dense_shape[1], max_line_len)
            self.assertEqual(trg.dense_shape[1], max_line_len + 2)







if __name__ == '__main__':
    unittest.main()

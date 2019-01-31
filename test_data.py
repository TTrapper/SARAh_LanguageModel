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
            self.assertEqual(trg.dense_shape[2], max_word_len + 2) # +2 for GO/STOP

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
            self.assertEqual(trg.dense_shape[1], max_line_len + 2) # +2 for GO/STOP


class TestData(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 20
            max_line_len = 64
            basedir = './example_data/train_and_eval/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.traindir + '*')
            src, trg_word_enc, trg_word_dec, trg = sess.run([data.src, data.trg_word_enc, data.trg_word_dec, data.trg])

    def test_trg_follows_src(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 6
            max_word_len = 20
            max_line_len = 64
            basedir = './example_data/train_and_eval/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.traindir + '*')
            src, _, _, trg = sess.run([data.src, data.trg_word_enc, data.trg_word_dec, data.trg])
            print "**** TESTING THAT TRG LINES FOLLOW AFTER SRC LINES IN DATASET ****"
            for src_str, trg_str in zip(data.array_to_strings(src), data.array_to_strings(trg)):
                src_file, src_line = self.find_line_in_dataset(data.traindir, src_str)
                print src_str
                if src_file is None:
                    print "Either this src line wasn't found or isn't unique"
                    print
                    continue
                line_after_src =  open(data.traindir + src_file).readlines()[src_line + 1].strip()
                print line_after_src
                cleaned_trg = trg_str.replace(data.go_stop_token, '').strip()
                print cleaned_trg
                print
                self.assertEqual(line_after_src, cleaned_trg)
            print "**************************************************"

    def find_line_in_dataset(self, datadir, line):
        line = line.strip()
        for fname in os.listdir(datadir):
            lines = open(datadir + fname).readlines()
            lines = [l.strip() for l in lines]
            # only return index if line is unique in the file
            indices = [i for i, file_line in enumerate(lines) if line == file_line]
            if len(indices) == 1:
                return fname, indices[0]
        return None, -1

    def test_manual_inspection(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 6
            max_word_len = 1024
            max_line_len = 1024
            basedir = './example_data/train_and_eval/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.traindir + '*')
            src, trg_word_enc, trg_word_dec, trg = [data.array_to_strings(a) for a in
                sess.run([data.src, data.trg_word_enc, data.trg_word_dec, data.trg])]

            print "***** BEGIN MANUAL INSPECTION ******"
            for src, trg_word_enc, trg_word_dec, trg in zip(src, trg_word_enc, trg_word_dec, trg):
                print src
                print data_pipe.replace_pad_chrs(trg_word_enc)
                print data_pipe.replace_pad_chrs(trg_word_dec)
                print data_pipe.replace_pad_chrs(trg)
                print
            print "***** END MANUAL INSPECTION ******"

    def test_seq_lens(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 3
            max_word_len = 16
            max_line_len = 32
            basedir = './example_data/train_and_eval/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.traindir + '*')
            src, trg_word_enc, trg_word_dec, trg, src_sentence_len, trg_sentence_len, trg_word_len\
                = sess.run([data.src, data.trg_word_enc, data.trg_word_dec, data.trg,
                data.src_sentence_len, data.trg_sentence_len, data.trg_word_len])
            src, trg_word_enc, trg_word_dec, trg = [data.array_to_strings(a) for a in
                [src, trg_word_enc, trg_word_dec, trg]]

            for src_sentence, trg_sentence, src_sent_len, trg_sent_len, trg_wrd_len in zip(
                src, trg, src_sentence_len, trg_sentence_len, trg_word_len):
                for word, length in zip(trg_sentence.split(' '), trg_wrd_len):
                    self.assertEqual(len(word), length)
                self.assertEqual(len(src_sentence.strip().split(' ')), src_sent_len)
                self.assertEqual(len(trg_sentence.strip().split(' ')), trg_sent_len)




if __name__ == '__main__':
    unittest.main()

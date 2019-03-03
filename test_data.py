import os
import time
import unittest

import tensorflow as tf
import numpy as np

import data_pipe

def find_line_in_dataset(datadir, line):
    line = line.strip()
    for fname in os.listdir(datadir):
        lines = open(datadir + fname).readlines()
        indices = [i for i, file_line in enumerate(lines) if line == file_line.strip()]
        if len(indices) > 0:
            return fname, indices
    return fname, []

class TestPipeline(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            datadir = './example_data/'
            batch_size = 2
            unk_token = chr(1)
            file_cycle_len = len(os.listdir(datadir))
            _, _, chr_to_id = data_pipe.create_chr_dicts(datadir, unk_token)
            iterator, filepattern = data_pipe.make_pipeline(batch_size, chr_to_id, file_cycle_len,
                shuffle_buffer=16)
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer, feed_dict={filepattern:datadir + '*'})
            src_op, trg_op = iterator.get_next()
            src, trg = sess.run([src_op, trg_op])

    def test_trg_follows_src(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            datadir = './example_data/'
            batch_size = 7
            unk_token = chr(1)
            file_cycle_len = len(os.listdir(datadir))
            _, id_to_chr, chr_to_id = data_pipe.create_chr_dicts(datadir, unk_token)
            iterator, filepattern = data_pipe.make_pipeline(batch_size, chr_to_id, file_cycle_len,
                shuffle_buffer=16)
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer, feed_dict={filepattern:datadir + '*'})
            src_op, trg_op = iterator.get_next()
            for batch in range(4): # Do the check over a few batches
                src, trg = sess.run([src_op, trg_op])
                src = data_pipe.array_to_strings(src, id_to_chr)
                trg = data_pipe.array_to_strings(trg, id_to_chr)
                for src, trg in zip(src, trg):
                    # check lines can be found in dataset
                    src_filename, src_indices = find_line_in_dataset(datadir, src)
                    self.assertTrue(len(src_indices) > 0)
                    trg_filename, trg_indices = find_line_in_dataset(datadir, trg)
                    self.assertTrue(len(trg_indices) > 0)
                    # check trg lines follow from src (if the lines are unique)
                    if (len(src_indices) == 1 and len(trg_indices) == 1):
                        self.assertTrue(src_indices[0] + 1 == trg_indices[0])

    def test_inference_pipeline(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            datadir = './example_data/'
            src_line = 'This is a source line which should end up the same for both pipes .\n'
            trg_line = 'This is a target line which should also look the same regardless of pipe .'
            with open('./tmp_test_data', 'w') as tmp_data:
                tmp_data.write(src_line)
                tmp_data.write(trg_line)
            batch_size = 2 # applies to regular pipeline only.
            unk_token = chr(1)
            file_cycle_len = 1
            _, id_to_chr, chr_to_id = data_pipe.create_chr_dicts(datadir, unk_token, file_cycle_len)
            # file pipeline
            iterator, filepattern = data_pipe.make_pipeline(batch_size, chr_to_id, file_cycle_len,
                shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./tmp_test_data'})
            src_op, trg_op = iterator.get_next()
            file_ops = [src_op, trg_op]
            # placeholder pipeline
            src_place, trg_place, src_op, trg_op = data_pipe.make_inference_pipeline(chr_to_id)
            placeholder_ops = [src_op, trg_op]
            sess.run(tf.tables_initializer())
            placeholder_results = sess.run(placeholder_ops, feed_dict={src_place:src_line.strip(), trg_place:trg_line})
            file_results =  sess.run(file_ops)
            # Should get identical data representations from placeholder as from file
            for placepipe, filepipe in zip(file_results, placeholder_results):
                # filepipe has batch_size 2 with identical etries, placepipe is broadcasted
                self.assertTrue(np.equal(placepipe, filepipe).all())

class TestData(unittest.TestCase):
    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 20
            max_line_len = 64
            basedir = './example_data/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.datadir + '*')
            src, trg_word_enc, trg_word_dec, trg = sess.run([data.src, data.trg_word_enc, data.trg_word_dec, data.trg])

    def test_trg_follows_src(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 6
            max_word_len = 20
            max_line_len = 64
            basedir = './example_data/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.datadir + '*')
            src, _, _, trg = sess.run([data.src, data.trg_word_enc, data.trg_word_dec, data.trg])
            print "**** TESTING THAT TRG LINES FOLLOW AFTER SRC LINES IN DATASET ****"
            for src_str, trg_str in zip(data.array_to_strings(src), data.array_to_strings(trg)):
                src_file, src_line = self.find_line_in_dataset(data.datadir, src_str)
                print src_str
                if src_file is None:
                    print "Either this src line wasn't found or isn't unique"
                    print
                    continue
                line_after_src =  open(data.datadir + src_file).readlines()[src_line + 1].strip()
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
            basedir = './example_data/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.datadir + '*')
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
            basedir = './example_data/'
            data = data_pipe.Data(basedir, batch_size, max_word_len, max_line_len)
            sess.run(tf.tables_initializer())
            data.initialize(sess, data.datadir + '*')
            (src,
             trg_word_enc,
             trg_word_dec,
             trg,
             src_sentence_len,
             trg_sentence_len,
             trg_word_len) = sess.run([data.src,
                                       data.trg_word_enc,
                                       data.trg_word_dec,
                                       data.trg,
                                       data.src_sentence_len,
                                       data.trg_sentence_len,
                                       data.trg_word_len])
            [src,
             trg_word_enc,
             trg_word_dec,
             trg] = [data.array_to_strings(a) for a in [src, trg_word_enc, trg_word_dec, trg]]

            for src_sentence, trg_sentence, src_sent_len, trg_sent_len, trg_wrd_len in zip(
                src, trg, src_sentence_len, trg_sentence_len, trg_word_len):
                for word, length in zip(trg_sentence.split(' '), trg_wrd_len):
                    self.assertEqual(len(word), length)
                self.assertEqual(len(src_sentence.strip().split(' ')), src_sent_len)
                self.assertEqual(len(trg_sentence.strip().split(' ')), trg_sent_len)




if __name__ == '__main__':
    unittest.main()

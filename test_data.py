import os
import unittest

import tensorflow as tf
import numpy as np

import prepare_data
import data_pipe

class TestPipeline(unittest.TestCase):
    def setup_char_to_id(self, sess):
        _, _, chr_to_id = data_pipe.create_chr_dicts('./example_data/', chr(0), chr(1))
        sess.run(tf.tables_initializer())
        return chr_to_id

    def test_compiles(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 20
            max_line_len = 64
            chr_to_id = self.setup_char_to_id(sess)
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                chr_to_id, shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            line_op = iterator.get_next()
            line = sess.run(line_op)

    def test_max_word_len(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 4
            max_line_len = 1024
            chr_to_id = self.setup_char_to_id(sess)
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                chr_to_id, shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            (line_val, line_row_lens) = iterator.get_next()
            line_op, _, _ = data_pipe._compose_ragged_batch(line_val, line_row_lens)
            line = sess.run(line_op.to_sparse())
            self.assertEqual(line.dense_shape[2], max_word_len + 1) # +1 for STOP

    def test_max_line_len(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 2
            max_word_len = 1024
            max_line_len = 4
            chr_to_id = self.setup_char_to_id(sess)
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                chr_to_id, shuffle_buffer=16)
            sess.run(iterator.initializer, feed_dict={filepattern:'./example_data/*.txt'})
            (line_val, line_row_lens) = iterator.get_next()
            line_op, _, _ = data_pipe._compose_ragged_batch(line_val, line_row_lens)
            line = sess.run(line_op.to_sparse())
            self.assertEqual(line.dense_shape[1], max_line_len + 1) # +1 for STOP

    def test_inference_pipeline(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            txt_line = 'This is a line of txt which should look the same regardless of pipeline used .'
            with open('./tmp_test_data', 'w') as tmp_data:
                tmp_data.write(txt_line)
            batch_size = 2 # applies to regular pipeline only.
            max_word_len = 20
            max_line_len = 32
            chr_to_id = self.setup_char_to_id(sess)
            # file pipeline
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                chr_to_id, shuffle_buffer=1)
            sess.run(iterator.initializer, feed_dict={filepattern:'./tmp_test_data'})
            (line_val, line_row_lens) = iterator.get_next()
            line_op, _, _ = data_pipe._compose_ragged_batch(line_val, line_row_lens)
            file_results = sess.run(line_op)
            # placeholder pipeline
            line_place, (line_val, line_row_lens) = data_pipe.make_inference_pipeline(chr_to_id)
            line_op, _, _ = data_pipe._compose_ragged_batch(line_val, line_row_lens)
            placeholder_results = sess.run(line_op, feed_dict={line_place:txt_line})
            # Should get identical data representations from placeholder as from file
            file_results = file_results.to_list()
            placeholder_results = placeholder_results.to_list()
            # file_results has batch_size 2 with identical entries, check both
            self.assertEqual(file_results[0], placeholder_results[0])
            self.assertEqual(file_results[1], placeholder_results[0])

    def test_shuffle_words(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_size = 6
            max_line_len = 128
            max_word_len = 3
            max_chr_id = 8
            # An artificial batch of sentence
            char_ids = tf.constant(np.random.randint(0, max_chr_id, [batch_size, max_line_len, max_word_len]))
            sentence_lens = tf.constant(np.random.randint(max_line_len/2, max_line_len, [batch_size]))
            char_ids_shuffled = data_pipe.shuffle_words(char_ids, sentence_lens)
            char_ids, sentence_lens, char_ids_shuffled = sess.run([char_ids, sentence_lens, char_ids_shuffled])
            for line, sent_len, line_shuffled in zip(char_ids, sentence_lens, char_ids_shuffled):
                # Pad words (the ones beyond the sentence length) should be untouched
                self.assertTrue((line[sent_len:] == line_shuffled[sent_len:]).all())
                # If the words of the sentence have been shuffled, then the shuffled line must differ
                self.assertFalse((line == line_shuffled).all())
                # Each word in the shuffled line appears in the original
                line = line[:sent_len]
                for shuffled_word in line[:sent_len]:
                    self.assertTrue(shuffled_word in line)

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
            trg = sess.run(data.trg)

    def find_line_in_dataset(self, datadir, line):
        """
        Finds all occurences of line contained in datadir. Returns dict mapping filenames
        to the line number where line was found.
        """
        line = line.strip()
        found = {}
        for fname in os.listdir(datadir):
            lines = open(datadir + fname).readlines()
            lines = [l.strip() for l in lines]
            indices = [i for i, file_line in enumerate(lines) if line == file_line]
            if len(indices) > 0:
                found[fname] = indices
        return found

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

            (trg,
             trg_sentence_len,
             trg_word_len) = sess.run([data.trg.to_tensor(-1),
                data.trg_sentence_len,
                data.trg_word_len.to_tensor(0)])
            trg = data.array_to_strings(trg)

            print "***** BEGIN MANUAL INSPECTION ******"
            for trg, trg_sentence_len, trg_word_len in zip(trg, trg_sentence_len, trg_word_len):
                print data_pipe.replace_pad_chrs(trg)
                print trg_sentence_len
                print trg_word_len
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
            (trg,
             trg_sentence_len,
             trg_word_len) = sess.run([data.trg.to_tensor(default_value=-1),
                                       data.trg_sentence_len,
                                       data.trg_word_len.to_tensor()])
            trg = data.array_to_strings(trg)
            for trg_sentence, trg_sent_len, trg_wrd_len in zip(trg, trg_sentence_len, trg_word_len):
                for word, length in zip(trg_sentence.split(' '), trg_wrd_len):
                    self.assertEqual(len(word), length)
                self.assertEqual(len(trg_sentence.strip().split(' ')), trg_sent_len)




if __name__ == '__main__':
    unittest.main()

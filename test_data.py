import os
import unittest

import tensorflow as tf
import numpy as np

import prepare_data
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
            self.assertEqual(src.dense_shape[2], max_word_len + 1)
            self.assertEqual(trg.dense_shape[2], max_word_len + 1) # +1 for STOP

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
            self.assertEqual(src.dense_shape[1], max_line_len + 1)
            self.assertEqual(trg.dense_shape[1], max_line_len + 1) # +1 for STOP

    def test_inference_pipeline(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            src_line = 'This is a source line which should end up the same for both pipes .\n'
            trg_line = 'This is a target line which should also look the same regardless of pipe .'
            with open('./tmp_test_data', 'w') as tmp_data:
                tmp_data.write(src_line)
                tmp_data.write(trg_line)

            batch_size = 2 # applies to regular pipeline only.
            max_word_len = 20
            max_line_len = 32
            go_stop_token = chr(0)
            unk_token = chr(1)
            _, _, chr_to_id = data_pipe.create_chr_dicts('./example_data/',
                go_stop_token, unk_token)
            # file pipeline
            iterator, filepattern = data_pipe.make_pipeline(batch_size, max_word_len, max_line_len,
                shuffle_buffer=1)
            sess.run(iterator.initializer, feed_dict={filepattern:'./tmp_test_data'})
            src_op, trg_op = iterator.get_next()
            file_op = data_pipe.sparse_chr_to_dense_id(chr_to_id, src_op, trg_op)
            # placeholder pipeline
            src_place, trg_place, src_op, trg_op = data_pipe.make_inference_pipeline()
            placeholder_ops = data_pipe.sparse_chr_to_dense_id(chr_to_id, src_op, trg_op)
            sess.run(tf.tables_initializer())
            placeholder_results = sess.run(placeholder_ops, feed_dict={src_place:src_line.strip(), trg_place:trg_line})
            file_results =  sess.run(file_op)
            # Should get identical data representations from placeholder as from file
            for placepipe, filepipe in zip(file_results, placeholder_results):
                # filepipe has batch_size 2 with identical etries, placepipe is broadcasted
                self.assertTrue(np.equal(placepipe, filepipe).all()) #

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
            src, trg = sess.run([data.src, data.trg])

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
            src, trg = sess.run([data.src, data.trg])
            print "**** TESTING THAT TRG LINES FOLLOW AFTER SRC LINES IN DATASET ****"
            not_found_count = 0
            for src_str, trg_str in zip(data.array_to_strings(src), data.array_to_strings(trg)):
                cleaned_src = src_str.replace(data.go_stop_token, '').strip()
                print cleaned_src
                cleaned_trg = trg_str.replace(data.go_stop_token, '').strip()
                print cleaned_trg
                file_line_map = self.find_line_in_dataset(data.datadir, cleaned_src)
                if len(file_line_map) == 0:
                    print "WARNING: src line was not found in dataset, the input pipeline likely modified it"
                    print
                    not_found_count += 1
                    continue
                num_matches = 0
                for fname, indices in file_line_map.iteritems():
                    for index in indices:
                        line_after_src =  open(data.datadir + fname).readlines()[index + 1].strip()

                        line_after_src = sess.run(data.trg_inference, feed_dict={data.trg_place: line_after_src})
                        line_after_src = data.array_to_strings(line_after_src)[0]
                        line_after_src = line_after_src.replace(data.go_stop_token, '').strip()
                        print line_after_src
                        if line_after_src == cleaned_trg:
                            num_matches += 1
                print
                # At least one of the src lines found was followed by the trg line
                self.assertTrue(num_matches >= 1)
            self.assertTrue(not_found_count < batch_size) # At least one source line was found
            print "**************************************************"

    def find_line_in_dataset(self, datadir, line):
        """
        Finds all occurences of line files contained in datadir. Returns dict mapping filenames
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
            src, trg = [data.array_to_strings(a) for a in sess.run([data.src, data.trg])]

            print "***** BEGIN MANUAL INSPECTION ******"
            for src, trg in zip(src, trg):
                print data_pipe.replace_pad_chrs(src)
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
             trg,
             src_sentence_len,
             trg_sentence_len,
             trg_word_len) = sess.run([data.src,
                                       data.trg,
                                       data.src_sentence_len,
                                       data.trg_sentence_len,
                                       data.trg_word_len])
            [src, trg] = [data.array_to_strings(a) for a in [src, trg]]

            for src_sentence, trg_sentence, src_sent_len, trg_sent_len, trg_wrd_len in zip(
                src, trg, src_sentence_len, trg_sentence_len, trg_word_len):
                for word, length in zip(trg_sentence.split(' '), trg_wrd_len):
                    self.assertEqual(len(word), length)
                self.assertEqual(len(src_sentence.strip().split(' ')), src_sent_len)
                self.assertEqual(len(trg_sentence.strip().split(' ')), trg_sent_len)




if __name__ == '__main__':
    unittest.main()

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
    def test_inference_vs_train(self):
        self.assertTrue(False) # disable and auto fail this test for now
        tf.reset_default_graph()
        with tf.Session() as sess:
            conf = config.generate_config(keep_prob=1.0)
            conf['batch_size'] = 1
            data = data_pipe.Data('./example_data/', conf['batch_size'], conf['max_word_len'], conf['max_line_len'])
            model, free_model = train.build_model(data, conf)
            data.initialize(sess, data.datadir + '*')
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            (out_logits_4,
             src_sentence_3,
             src_sent_len_1,
             trg_sentence_3,
             trg_sent_len_1) = sess.run([model.out_logits_4,
                                         data.src.to_tensor(-1),
                                         data.src_sentence_len,
                                         data.trg.to_tensor(-1),
                                         data.trg_sentence_len])
            src = data.array_to_strings(src_sentence_3)[0].replace(data.go_stop_token, '')
            trg = data.array_to_strings(trg_sentence_3)[0].replace(data.go_stop_token, '')
            # trg is the concatenation of itself with src. Restore the stop word that delimits them
            trg = trg[len(src):]
            trg = src + ' ' + data.go_stop_token + ' ' + trg.strip() # recombine src and trg
            print src
            print trg
            feed = {data.src_place:src, data.trg_place:trg}
            (free_logits_4,
             src_sentence_inference,
             trg_sentence_inference) = sess.run([free_model.out_logits_4,
                                                 data.src_inference.to_tensor(-1),
                                                 data.trg_inference.to_tensor(-1)], feed_dict=feed)
            # Get the fist batch line and trim potential batch padding from the model's logits
            out_logits_3 = out_logits_4[0, :free_logits_4.shape[1], :free_logits_4.shape[2], :]
            # Check that the model's outputs are the same regardless of what data pipeline is used
            self.assertTrue((np.abs(out_logits_3 - free_logits_4[0]) < 1e-5).all())
            # Run the inference model as though generating one char at time, and check the outputs
            feed = {data.src_place:src, data.trg_place:''} # Start with no input
            free_logits_4 = sess.run(free_model.out_logits_4, feed_dict=feed)
            self.assertTrue((np.abs(free_logits_4[0,0,0,:] - out_logits_3[0,0,:]) <= 1e-5).all()) 
            trg = trg.split()
            trg_so_far = ''
            for word_idx, trg_word in enumerate(trg):
                for chr_num in range(len(trg_word)):
                    trg_so_far += trg_word[chr_num]
                    feed = {data.src_place:src, data.trg_place:trg_so_far}
                    free_logits_4 = sess.run(free_model.out_logits_4, feed_dict=feed)
#                    print (free_logits_4[0, word_idx, chr_num + 1,:] - out_logits_3[word_idx, chr_num + 1, :]) < 1e-4
                    self.assertTrue((np.abs(free_logits_4[0, word_idx, chr_num + 1,:] - out_logits_3[word_idx, chr_num + 1, :]) <= 1e-5).all())
                trg_so_far += ' '



if __name__ == '__main__':
    unittest.main()

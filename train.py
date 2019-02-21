import argparse
from collections import deque

import numpy as np
import tensorflow as tf

import data_pipe
import model_def
import config

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
args = parser.parse_args()

def build_model(data, conf):
    with tf.variable_scope('Model', dtype=tf.float32):
        model = model_def.Model(data.src,
            data.src_sentence_len,
            data.trg_word_enc,
            data.trg_sentence_len,
            data.trg_word_dec,
            data.trg_word_len,
            len(data.chr_to_freq),
            conf)
    return model

def calc_loss(logits, targets, word_lens, sentence_lens):
    mask = tf.where(tf.equal(targets, -1), tf.zeros_like(targets), tf.ones_like(targets))
    targets *= mask # softmax doesn't like oov targets, turn -1 padding to 0
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss *= tf.cast(mask, loss.dtype) # remove loss calculated over the padding chars
    # Don't divide by zero
    word_lens = tf.where(tf.equal(word_lens, 0), tf.ones_like(word_lens), word_lens)
    sentence_lens = tf.where(tf.equal(sentence_lens, 0), tf.ones_like(sentence_lens), sentence_lens)
    # Get cost per char, account for varying sequence lengths
    loss = tf.reshape(loss, tf.shape(targets))
    loss = tf.reduce_sum(loss, axis=2) / tf.cast(word_lens, loss.dtype)
    loss = tf.reduce_sum(loss, axis=1) / tf.cast(sentence_lens, loss.dtype)
    loss = tf.reduce_mean(loss) # batch mean
    return loss

def build_train_op(loss, learn_rate, max_grad_norm):
    tvars = tf.trainable_variables()
    grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learn_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return train_op, grads, norm

def train():
    conf = config.config
    data = data_pipe.Data(args.datadir, conf['batch_size'], conf['max_word_len'], conf['max_line_len'])
    model = build_model(data, conf)
    loss = calc_loss(model.out_logits_4, data.trg, data.trg_word_len, data.trg_sentence_len)
    train_op, grads, norm = build_train_op(loss, conf['learn_rate'], conf['max_grad_norm'])
    sess = tf.Session()
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    data.initialize(sess, data.traindir + '*')
    recent_costs = deque(maxlen=100)
    while True:
        _, g, n, c, out_logits_4, trg, trg_word_dec, trg_word_enc = sess.run(
            [train_op,
                grads,
                norm,
                loss,
                model.out_logits_4,
                data.trg,
                data.trg_word_dec,
                data.trg_word_enc])

        recent_costs.append(c)
        print sum(recent_costs)/len(recent_costs), n

#        print trg_word_enc.shape
#        print data_pipe.replace_pad_chrs('\n'.join(data.array_to_strings(trg_word_enc)))
#        print trg_word_dec.shape
#        print data_pipe.replace_pad_chrs('\n'.join(data.array_to_strings(trg_word_dec)))
#        print trg.shape
#        print data_pipe.replace_pad_chrs('\n'.join(data.array_to_strings(trg)))
#        print trg
#        out_logits_4 = np.argmax(out_logits_4, axis=3)
#        print out_logits_4
#        print


train()

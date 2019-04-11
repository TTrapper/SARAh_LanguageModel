import argparse
from collections import deque
import os
import sys

import numpy as np
import tensorflow as tf

import data_pipe
import model_def
import config

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--restore', type=str, default=None)
parser.add_argument('--keep_prob', type=float, default=0.9)

def build_model(data, conf, reuse=False):
    with tf.variable_scope('Model', dtype=tf.float32, reuse=reuse) as scope:
        model = model_def.Model(data.src,
            data.src_sentence_len,
            data.trg,
            data.trg_sentence_len,
            len(data.chr_to_freq),
            conf)
        scope.reuse_variables()
        # Create a copy of the model that operates over the inference pipeline
        conf = config.generate_config(1.0)
        free_model = model_def.Model(data.src_inference,
            data.src_sentence_len_inference,
            data.trg_inference,
            data.trg_sentence_len_inference,
            len(data.chr_to_freq),
            conf)
    for v in tf.trainable_variables():
        print v
    return model, free_model

def calc_loss(logits, targets, word_lens, sentence_lens):
    word_lens = word_lens.to_tensor(0)
    targets = targets.to_tensor(-1)
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

def run_inference(model, data, conf, sess):
    paths = [args.datadir + p for p in os.listdir(args.datadir)]
    conditions = data_pipe.getRandomSentence(paths, numSamples=2)
    for condition in conditions:
        condition = condition[0].strip()
        print condition
        for softmax_temp in [1e-16, 0.75]:
            print softmax_temp
            # Reconstruct src from encoder representation
            run_inference_once(model, data, conf, sess, softmax_temp, condition, '')
            # Run inference for next sentence
            run_inference_once(model, data, conf, sess, softmax_temp, condition, condition)
        print

def run_inference_once(model, data, conf, sess, softmax_temp, src_condition, trg_condition):
    """
    src_condition: string representing the source sentence
    trg_condition: string representing the source sentence, or an empty string. The model is
        trained to first reconstruct the source sentence, then predict the target. Providing
        the source sentence here allows inference to skip straight to generating the target.
    """
    # TODO: This is very slow because the entire model is rerun for each char prediction
    result = trg_condition
    char_idx = 0
    word_idx = len(result.split())
    try:
        for char_count in range(128):
            feed = {data.src_place:src_condition, data.trg_place:result.strip(), model.softmax_temp:softmax_temp}
            predictions = sess.run(model.predictions_3, feed_dict=feed)
            next_char = data.id_to_chr[predictions[0, word_idx, char_idx]]
            if next_char == data.go_stop_token:
                if char_idx == 0: # EOS
                    result += ' ' + data.go_stop_token
                result += ' '
                word_idx += 1
                char_idx = 0
            else:
                result += next_char
                char_idx += 1
        result = result.replace(trg_condition, '').replace(' {} '.format(data.go_stop_token), ' __ ')
        print result
    except Exception as e:
        print 'Inference failed: '
        print e
        print result

def train():
    conf = config.generate_config(args.keep_prob)
    data = data_pipe.Data(args.datadir, conf['batch_size'], conf['max_word_len'], conf['max_line_len'])
    model, free_model = build_model(data, conf)
    loss = calc_loss(model.out_logits_4, data.trg, data.trg_word_len, data.trg_sentence_len)
    train_op, grads, norm = build_train_op(loss, conf['learn_rate'], conf['max_grad_norm'])
    sess = tf.Session()
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    if args.restore is not None:
        saver.restore(sess, args.restore)
    data.initialize(sess, data.datadir + '*')
    recent_costs = deque(maxlen=100)
    batch_num = 0
    while True:
        [_,n,c] = sess.run([train_op,norm,loss])
        recent_costs.append(c)
        if batch_num%250 == 0:
            print batch_num, sum(recent_costs)/len(recent_costs), n
            saver.save(sess, './saves/model.ckpt')
        if batch_num%1000 == 10:
            run_inference(free_model, data, conf, sess)
            print
        sys.stdout.flush()
        batch_num += 1


if __name__ == '__main__':
    args = parser.parse_args()
    train()

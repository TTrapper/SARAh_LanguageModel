import argparse
from collections import deque
import os
import sys

import numpy as np
import tensorflow as tf

import data_pipe
import model_def
import config


def build_model(data, conf, reuse=False):
    with tf.variable_scope('Model', dtype=tf.float32, reuse=reuse) as scope:
        model = model_def.Model(data.src, data.trg, len(data.chr_to_freq), 0, conf)
        scope.reuse_variables()
        # Create a copy of the model that operates over the inference pipeline
        conf = config.get_config(keep_prob=1.0)
        free_model = model_def.Model(data.free_src, data.free_trg, len(data.chr_to_freq), 0, conf)
    return model, free_model

def calc_loss(logits, targets):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = tf.reduce_mean(loss)
    return loss

def build_train_op(loss, learn_rate, max_grad_norm):
    tvars = tf.trainable_variables()
    grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learn_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return train_op, grads, norm

def run_inference(model, data, conf, sess, softmax_temp=1e-24):
    # TODO: This is very slow because the entire model is rerun for each char prediction
    paths = [args.datadir + p for p in os.listdir(args.datadir)]
    condition = data_pipe.getRandomSentence(paths, numSamples=1, sampleRange=1)[0][0]
    condition = condition[:-1] # remove newline
    print condition
    result = ''
    try:
        for char_count in range(len(condition)):
            # The number of chars must be a multiple of chrs_per_word. Pad the result
            padlen = conf['chrs_per_word'] - (len(result) % conf['chrs_per_word'])
            padstr = ''.ljust(padlen)
            feed = {data.src_place:condition,
                    data.trg_place:result + padstr,
                    model.softmax_temp:softmax_temp}
            predictions = sess.run(model.predictions_2, feed_dict=feed)
            next_char = data.id_to_chr[predictions[0, char_count]]
            result += next_char
        print result
    except Exception as e:
        print 'Inference failed: '
        print e
        print result

def train():
    conf = config.get_config()
    data = data_pipe.Data(args.datadir, conf['batch_size'])
    model, free_model = build_model(data, conf)
    loss = calc_loss(model.out_logits_3, data.trg)
    train_op, grads, norm = build_train_op(loss, conf['learn_rate'], conf['max_grad_norm'])
    sess = tf.Session()
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    if args.restore_checkpoint is not None:
        print 'restoring checkpoint: ' +  args.restore_checkpoint
        saver.restore(sess, args.restore_checkpoint)
    data.initialize(sess, data.datadir + '*')
    recent_costs = deque(maxlen=100)
    batch_num = 0
    while True:
        [_,n,c] = sess.run([train_op,norm,loss])
        recent_costs.append(c)
        if batch_num%25 == 0:
            print batch_num, sum(recent_costs)/len(recent_costs), n
            saver.save(sess, './saves/model.ckpt')
            run_inference(free_model, data, conf, sess)
            sys.stdout.flush()
        batch_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--restore_checkpoint', type=str, default=None)
    args = parser.parse_args()
    train()

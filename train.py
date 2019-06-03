import argparse
from collections import deque
import os
import sys

import numpy as np
import tensorflow as tf

import data_pipe
import inference
import model_def
import config

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True,
    help='A directory containing txt files with one training example per line')
parser.add_argument('--inference_dir', type=str, default=None,
    help='A directory with txt files from which to sample random context lines for inference')
parser.add_argument('--restore', type=str, default=None,
    help='Path to a TF checkpoint to restore')
parser.add_argument('--keep_prob', type=float, default=1.0,
    help='Keep prob for dropout')
parser.add_argument('--noise_level', type=float, default=0.0,
    help='How much gaussian noise (proportional to layer\'s moments) to add to activations')
parser.add_argument('--eval_mode', type=str, default='no', choices=['yes','no','true','false'],
    help='If true (yes), disable training and aggragate cost over one epoch of data')
parser.add_argument('--train_words', type=str, default='yes', choices=['yes','no','true','false'],
    help='enable/disable training of word encoder and word decoder')
parser.add_argument('--inference_mode', type=str, default='no', choices=['yes','no','true','false'],
    help='If true (yes), disables training and runs inference')
parser.add_argument('--groundtruth_mode', type=str, default='half', choices=['half', '|SEP|', 'off'],
    help='Inference: determines how to split the context to display a ground_truth')

def parse_bool_arg(arg_str):
    return arg_str == 'yes' or arg_str == 'true'

def build_model(data, conf):
    with tf.variable_scope('Model', dtype=tf.float32) as scope:
        model = model_def.Model(data.trg,
            data.trg_sentence_len,
            len(data.id_to_chr),
            conf)
        scope.reuse_variables()
        # Create a copy of the model that operates over the inference pipeline
        conf = config.generate_config(keep_prob=1.0, noise_level=0.0)
        free_model = model_def.Model(
            data.trg_inference,
            data.trg_sentence_len_inference,
            len(data.id_to_chr),
            conf,
            inference_mode=True)
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

def get_vars_to_train(train_words):
    if train_words:
        trained_variables = tf.trainable_variables()
    else:
        trained_variables = [v for v in tf.trainable_variables() if 'word' not in v.name]
    print 'NOT-TRAINED VARIABLES:'
    for v in tf.trainable_variables():
        if v not in trained_variables:
            print v
    return trained_variables

def build_train_op(loss, learn_rate, max_grad_norm, vars_to_train):
    grads, norm = tf.clip_by_global_norm(tf.gradients(loss, vars_to_train), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learn_rate)
    train_op = optimizer.apply_gradients(zip(grads, vars_to_train))
    return train_op, grads, norm

def train_loop(sess, ops, saver, is_eval_mode, inferencer):
    recent_costs = deque(maxlen=100)
    batch_num = 0
    while True:
        batch_num += 1
        try:
            out = sess.run(ops)
            n = out['norm']
            c = out['loss']
            recent_costs.append(c)
            if is_eval_mode: # for eval we want the total cost over the entire epoch
                recent_costs = [sum(recent_costs)]
        except tf.errors.OutOfRangeError as e:
            print 'EOE:', batch_num, sum(recent_costs)/batch_num
            return
        if batch_num % 250 == 1:
            cost_window = len(recent_costs) if not is_eval_mode else batch_num
            print batch_num, sum(recent_costs)/cost_window, n
            if not is_eval_mode:
                saver.save(sess, './saves/model.ckpt')
        if not is_eval_mode and batch_num % 1000 == 10:
            inferencer.run_inference()
            print
        sys.stdout.flush()

def train():
    inference_dir = args.datadir if args.inference_dir is None else args.inference_dir
    conf = config.generate_config(args.keep_prob, args.noise_level)
    data = data_pipe.Data(args.datadir, conf['batch_size'], conf['max_word_len'],
        conf['max_line_len'], eval_mode=args.eval_mode)
    model, free_model = build_model(data, conf)
    loss = calc_loss(model.out_logits_4, data.trg, data.trg_word_len, data.trg_sentence_len)
    vars_to_train = get_vars_to_train(args.train_words)
    train_op, grads, norm = build_train_op(loss, conf['learn_rate'], conf['max_grad_norm'],
        vars_to_train)
    sess = tf.Session()
    inferencer = inference.InferenceRunner(sess,
                                           free_model,
                                           data,
                                           inference_dir,
                                           context_placeholder=data.trg_place,
                                           max_context_len=conf['max_line_len'] - 1,
                                           groundtruth_mode=args.groundtruth_mode,
                                           softmax_temps=[1e-16, 0.5, 0.75, 1.0],
                                           num_predictions=32,
                                           num_runs_per_call=2)
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    if args.restore is not None:
        restore_vars = tf.trainable_variables()
        restorer = tf.train.Saver(restore_vars, max_to_keep=1)
        print 'NOT-RESTORED VARIABLES:'
        for v in tf.trainable_variables():
            if v not in restore_vars:
                print v
        restorer.restore(sess, args.restore)
    if args.inference_mode:
        inferencer.run_inference()
        exit()
    data.initialize(sess, data.datadir + '*')
    ops = {'norm':norm, 'loss':loss}
    if not args.eval_mode:
        ops['train'] = train_op
    train_loop(sess, ops, saver, args.eval_mode, inferencer)

if __name__ == '__main__':
    args = parser.parse_args()
    args.eval_mode = parse_bool_arg(args.eval_mode)
    args.train_words = parse_bool_arg(args.train_words)
    args.inference_mode = parse_bool_arg(args.inference_mode)

    train()

import argparse
import os
import sys
import random

from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf

import words

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--restore', type=str, required=True)
parser.add_argument('--savename', type=str, default='./embeded_sentences')

def get_sentence_embeds(savename, datadir, restore):
    embed_file = '{}.embeds.npy'.format(savename)
    if os.path.exists(embed_file):
        print 'loading precomputed embeddings: {}'.format(embed_file)
        sentence_embeds = np.load(embed_file)
        labels = open('{}.labels'.format(savename)).readlines()
        return sentence_embeds, labels
    sentence_embeds, labels = compute_sentence_embeds(datadir, restore, savename)
    return sentence_embeds, labels

def run_loop(sess, data, sentence_embeds_2_op, sentences_3_op):
    batch_num = 0
    embeds = []
    labels = []
    while True:
        batch_num += 1
        try:
            sentence_embeds_2, sentences_3 = sess.run([sentence_embeds_2_op, sentences_3_op])
            sentences = data.array_to_strings(sentences_3, remove_stops=True)
            embeds.append(sentence_embeds_2)
            labels.extend(sentences)
        except tf.errors.OutOfRangeError as e:
            print '{} sentences encoded'.format(len(labels))
            break
        if batch_num%50 == 1:
            print '{} sentences encoded'.format(len(labels))
        sys.stdout.flush()
    return np.concatenate(embeds, axis=0), labels

def compute_sentence_embeds(datadir, restore, savename):
    print 'computing sentence embeddings from model: {}'.format(restore)
    model, _, data, sess = words.sess_setup(datadir, restore, batch_size=256)
    embeds, labels = run_loop(sess, data, model.sentence_embeds_2, data.trg.to_tensor(-1))
    print embeds.shape
    np.save('{}.embeds.npy'.format(savename), embeds)
    with open('{}.labels'.format(savename), 'w') as label_file:
        label_file.writelines(labels)
    return embeds, labels

    sentence_embeds = []
    sentence_labels = []
    for filename in os.listdir(datadir):
        sentences = open('{}/{}'.format(datadir, filename)).readlines()
        sentence_labels.extend(sentences)
        for batch_num, sentence in sentences:
            sentence_embeds_2 = sess.run(free_model.sentence_embeds_2, feed_dict={data.trg_place:sentence})
            sentence_embeds.append(sentence_embeds_2)
            if len(sentence_embeds) % 1000 == 1:
                print len(sentence_embeds)
    sentence_embeds = np.concatenate(sentence_embeds, axis=0)
    print sentence_embeds.shape
    np.save('{}.embeds.npy'.format(savename), sentence_embeds)
    with open('{}.labels'.format(savename), 'w') as label_file:
        label_file.writelines(sentence_labels)
    return sentence_embeds, sentence_labels

def show_similar(embeds, labels):
    kv = KeyedVectors(embeds.shape[-1])
    kv.add(labels, embeds)
    random_labels = random.sample(labels, 100)
    for label in random_labels:
        print label
        for (distance, sentence) in kv.most_similar(label):
            print distance, sentence
        print '--------------------------------------------------'

if __name__ == '__main__':
    args = parser.parse_args()
    sentence_embeds, labels = get_sentence_embeds(args.savename, args.datadir, args.restore)
    projected_embeds = words.project_embeds(args.savename, sentence_embeds)
    words.plot_projections(projected_embeds, labels, args.savename)
    show_similar(sentence_embeds, labels)



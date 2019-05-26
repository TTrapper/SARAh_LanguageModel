import argparse
import os
import sys
import random

from sklearn.metrics import pairwise_distances
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf

import words

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--restore', type=str, required=True)
parser.add_argument('--savename', type=str, default='./embeded_sentences')
parser.add_argument('--max_num', type=int, default=None)
parser.add_argument('--write_examples', type=str, default='no', choices=['yes', 'no'])

def get_sentence_embeds(savename, datadir, restore, max_num):
    embed_file = '{}.embeds.npy'.format(savename)
    if os.path.exists(embed_file):
        print 'loading precomputed embeddings: {}'.format(embed_file)
        sentence_embeds = np.load(embed_file)
        labels = open('{}.labels'.format(savename)).readlines()
        labels = [l.strip() for l in labels]
        return sentence_embeds, labels
    sentence_embeds, labels = compute_sentence_embeds(datadir, restore, savename, max_num)
    return sentence_embeds, labels

def run_loop(sess, data, sentence_embeds_2_op, sentences_3_op, max_num):
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
            sys.stdout.flush()
            break
        if max_num is not None and len(labels) >= max_num:
            print '{} sentences encoded'.format(len(labels))
            sys.stdout.flush()
            break
        if batch_num%50 == 1:
            print '{} sentences encoded'.format(len(labels))
            sys.stdout.flush()
    return np.concatenate(embeds, axis=0), labels

def compute_sentence_embeds(datadir, restore, savename, max_num):
    print 'computing sentence embeddings from model: {}'.format(restore)
    model, _, data, sess = words.sess_setup(datadir, restore, batch_size=256)
    embeds, labels = run_loop(sess, data, model.sentence_embeds_2, data.trg.to_tensor(-1), max_num)
    print embeds.shape
    np.save('{}.embeds.npy'.format(savename), embeds)
    with open('{}.labels'.format(savename), 'w') as label_file:
        label_file.write('\n'.join(labels))
    return embeds, labels

def get_paraphrases(embeds, labels, min_distant=1, max_distance=9):
    """
    experimental - finds pairs of sentences whose distance falls within a given range. The hope is
    to filter for pairs that have a minimum level of semantic similarity yet arent duplicates.
    But sometimes the spatially closest sentence is not the best semantic match.
    """
    for label, vector in zip(labels, embeds):
        dists = pairwise_distances([vector], embeds, 'euclidean')[0]
        dists[np.argmin(dists)] = np.inf # nearest should be itself, replace it to get next-nearest
        if np.min(dists) < max_distance and np.min(dists) > min_distant:
            print dists[np.argmin(dists)]
            print label
            print labels[np.argmin(dists)]
            print '--------------------------------------------------'
    return

def make_similars_examples(embeds, labels, n_nearby, savename):
    """
    Runs through each sentence in labels and finds the n_nearby closest sentences, writing each
    pair to a file that can be used as a set of examples for training. Each line of the file
    contains a pair of similar sentences separated by: '|SEP|'
    """
    filename = '{}.similars'.format(savename)
    print 'Writing examples to file: {}'.format(filename)
    with open(filename, 'w') as out_file:
        for num_processed, query_sentence in enumerate(labels):
            index = labels.index(query_sentence)
            vector = embeds[index, :]
            dists = pairwise_distances([vector], embeds, 'euclidean')[0]
            # try to skip all of the duplicates by removing the really short distances
            dists = [d if d > 1.0 else np.inf for d in dists]
            for _ in range(n_nearby):
                idx = np.argmin(dists)
                dist = dists[idx]
                close_sentence = labels[idx]
                out_file.write('{} |SEP| {}\n'.format(query_sentence, close_sentence))
                dists[np.argmin(dists)] = np.inf # replace this dist with inf before next iteration
            if num_processed % 500 == 0:
                print '{} lines processed'.format(num_processed + 1)
                sys.stdout.flush()

def show_similar(embeds, label, n_examples=100, n_nearby=4):
    for query_sentence in random.sample(labels, n_examples):
        index = labels.index(query_sentence)
        print index, query_sentence
        vector = embeds[index, :]
        dists = pairwise_distances([vector], embeds, 'euclidean')[0]
        for _ in range(n_nearby):
            idx = np.argmin(dists)
            dist = dists[idx]
            close_sentence = labels[idx]
            print dist, close_sentence
            dists[np.argmin(dists)] = np.inf # replace closest with inf to get next-nearest
        print '--------------------------------------------------'
    return

    # TODO: gensim is much faster but doesn't handle duplicate entries very well
    kv = KeyedVectors(embeds.shape[-1])
    kv.add(labels, embeds)
    random_labels = random.sample(labels, 10)
    for label in random_labels:
        print label
        for tup in kv.most_similar(label):
            print tup
        print '--------------------------------------------------'

if __name__ == '__main__':
    args = parser.parse_args()
    args.write_examples = args.write_examples == 'yes'
    sentence_embeds, labels = get_sentence_embeds(args.savename, args.datadir, args.restore,
        args.max_num)
    show_similar(sentence_embeds, labels)
    projected_embeds = words.project_embeds(args.savename, sentence_embeds)
    words.plot_projections(projected_embeds, labels, args.savename)
    if args.write_examples:
        make_similars_examples(sentence_embeds, labels, n_nearby=2, savename=args.savename)


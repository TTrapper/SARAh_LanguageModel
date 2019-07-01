import argparse
import os
import time
import sys
import random

from sklearn.metrics import pairwise_distances
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf

import words

parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, required=True,
    help='path to file containing one sentence on each line (each line should be unique)')
parser.add_argument('--restore', type=str, required=False,
    help='path to pretrained language model')
parser.add_argument('--savename', type=str, default='./embeded_sentences',
    help='where to save the resulting sentence encodings')
parser.add_argument('--max_num', type=int, default=None,
    help='the maximum number of sentences to encode')
parser.add_argument('--write_examples', type=str, default='no', choices=['yes', 'no'],
    help='whether or not to write a file with tuples of similar (nearby) sentences on each line')
parser.add_argument('--project', type=str, default='no', choices=['yes', 'no'],
    help='whether or not to run project and plot embeddings in 2D')
parser.add_argument('--batch_size', type=int, default=256,
    help='batch_size to use when encoding sentences')

def get_sentence_embeds(savename, datadir, restore, max_num, batch_size):
    embed_file = '{}.embeds.npy'.format(savename)
    if os.path.exists(embed_file):
        print 'loading precomputed embeddings: {}'.format(embed_file)
        sentence_embeds = np.load(embed_file)
        return sentence_embeds
    sentence_embeds = compute_sentence_embeds(datadir, restore, savename, max_num, batch_size)
    return sentence_embeds

def run_loop(sess, data, sentence_embeds_2_op, sentences_3_op, max_num):
    batch_num = 0
    num_encoded = 0
    embeds = []
    start_time = time.time()
    while True:
        batch_num += 1
        try:
            sentence_embeds_2 = sess.run(sentence_embeds_2_op)
            num_encoded += sentence_embeds_2.shape[0]
            embeds.append(sentence_embeds_2)
        except tf.errors.OutOfRangeError as e:
            break
        if max_num is not None and num_encoded >= max_num:
            break
        if batch_num%5 == 1:
            sps = num_encoded/(time.time() - start_time)
            print '{} sentences encoded ({} sentences/sec)\r'.format(num_encoded, sps),
            sys.stdout.flush()
    print '{} sentences encoded'.format(num_encoded)
    sys.stdout.flush()
    return np.concatenate(embeds, axis=0)

def compute_sentence_embeds(datadir, restore, savename, max_num, batch_size):
    print 'computing sentence embeddings from model: {}'.format(restore)
    model, _, data, sess = words.sess_setup(datadir, restore, batch_size=batch_size)
    embeds = run_loop(sess, data, model.sentence_embeds_2, data.trg.to_tensor(-1), max_num)
    print embeds.shape
    np.save('{}.embeds.npy'.format(savename), embeds)
    return embeds

def make_similars_examples(embeds, labels, n_nearby, savename):
    """
    Runs through each sentence in labels and finds the n_nearby closest sentences, writing each
    tuple to a file that can be used as a set of examples for training. Each line of the file
    contains a tuple of similar sentences separated by: '|SEP|'
    """
    filename = '{}.similars'.format(savename)
    print 'Writing examples to file: {}'.format(filename)
    with open(filename, 'w') as out_file:
        batch_size = 1024
        num_processed = 0
        for batch in range(len(labels)/batch_size):
            query_sentences = labels[batch*batch_size:batch_size + batch*batch_size]
            vectors = embeds[batch*batch_size:batch_size + batch*batch_size]
            dists = pairwise_distances(vectors, embeds, 'euclidean')
            # Use argpartition to quickly get the indices of the closest n sentences
            nearby_ind = np.argpartition(dists, kth=1 + n_nearby, axis=1)[:, :1 + n_nearby]
            # Loop over the batch of sentences
            for ind_row, dist_row, query_sentence in zip(nearby_ind, dists, query_sentences):
                ind_row = ind_row[np.argsort(dist_row[ind_row])][1:] # sort, remove the query's idx
                similar_sentences = [query_sentence]
                similar_sentences.extend([labels[idx] for idx in ind_row])
                out_file.write(' |SEP| '.join(similar_sentences) + '\n')
            num_processed += batch_size
            if batch % 2 == 0:
                print '{} lines processed'.format(num_processed)
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
    args.project = args.project == 'yes'
    sentence_embeds = get_sentence_embeds(args.savename, args.labels, args.restore,
        args.max_num, args.batch_size)
    labels = open(args.labels).readlines()[:sentence_embeds.shape[0]]
    show_similar(sentence_embeds, labels)
    if args.write_examples:
        make_similars_examples(sentence_embeds, labels, n_nearby=4, savename=args.savename)
    if args.project:
        projected_embeds = words.project_embeds(args.savename, sentence_embeds)
        words.plot_projections(projected_embeds, labels, args.savename)

import argparse
import os
import time
import sys
import random

from sklearn.metrics import pairwise_distances_chunked
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

def run_loop(sess, save_path, shape_path, sentence_embeds_2_op, max_num):
    batch_num = 0
    num_encoded = 0
    embed_size = sentence_embeds_2_op.shape.as_list()[1]
    embeds = np.memmap(save_path, dtype=np.float16, mode='w+', shape=(1, embed_size))
    start_time = time.time()
    while True:
        try:
            sentence_embeds_2 = sess.run(sentence_embeds_2_op)
            batch_size = sentence_embeds_2.shape[0]
            num_encoded += batch_size
            embeds = np.memmap(save_path, dtype=np.float16, mode='r+', shape=(num_encoded, embed_size))
            embeds[batch_num*batch_size:, :] = sentence_embeds_2
        except tf.errors.OutOfRangeError as e:
            break
        except KeyboardInterrupt:
            print 'Interrupted. Saving shape of embeddings and exiting'
            np.save(shape_path, np.array([embeds.shape[0], embed_size]))
            exit()
        batch_num += 1
        if max_num is not None and num_encoded >= max_num:
            break
        if batch_num%5 == 1:
            sps = num_encoded/(time.time() - start_time)
            print '{} sentences encoded ({} sentences/sec)\r'.format(
                num_encoded, sps),
            sys.stdout.flush()
    print '{} sentences encoded'.format(num_encoded)
    np.save(shape_path, np.array([num_encoded, embed_size]))
    sys.stdout.flush()
    return embeds

def compute_sentence_embeds(datadir, restore, savename, max_num, batch_size):
    part_num = 0
    embed_path = '{}.dat'.format(savename)
    shape_path = '{}.shape.npy'.format(savename)
    if os.path.exists(embed_path):
        assert os.path.exists(shape_path)
        shape = tuple(np.load(shape_path))
        print 'Using precomputed embeddings: {} with shape {}'.format(embed_path, shape)
        embeds = np.memmap('{}.dat'.format(savename),  dtype=np.float16, mode='r', shape=shape)
        return embeds
    print 'computing sentence embeddings from model: {}'.format(restore)
    model, _, _, sess = words.sess_setup(datadir, restore, batch_size=batch_size)
    embeds = run_loop(sess, embed_path, shape_path, tf.cast(model.sentence_embeds_2, tf.float16), max_num)
    return embeds

def _make_nearest_k_fn(k):
    def nearest_k(dist, start):
        #FIXME breaks if run on a batch with < k vectors
        indices = np.argpartition(dist, kth=k, axis=1)[:, :k]
        row_indices = np.repeat(np.arange(dist.shape[0]), k)
        col_indies = np.reshape(indices, [-1])
        values = dist[row_indices, col_indies]
        values = np.reshape(values, [-1, k])
        return indices, values
    return nearest_k

def _find_nearest(query_vectors, embeds, n_nearby, batch_size):
    num_embeds = embeds.shape[0]
    num_batches = (num_embeds / batch_size) + (1 if num_embeds % batch_size != 0 else 0)
    nearest_k_fn = _make_nearest_k_fn(k=n_nearby)
    indices = []
    dists = []
    for b in range(num_batches):
        print 'Running batch {} of {}\r'.format(1 + b, num_batches),
        sys.stdout.flush()
        start = b * batch_size
        end = min(start + batch_size, embeds.shape[0])
        batch = embeds[start:end, :]
        batch_indices, batch_dists = zip(*pairwise_distances_chunked(query_vectors, batch,
            reduce_func=nearest_k_fn, metric='euclidean'))
        indices.append(np.concatenate(batch_indices) + start)
        dists.append(np.concatenate(batch_dists))
    # Combine results from each batch, sort them and take the n_nearby closest
    indices = np.concatenate(indices, axis=1)
    dists = np.concatenate(dists, axis=1)
    sorted_idx = np.argsort(dists, axis=1)[:, :n_nearby]
    indices = np.concatenate([[row[idx_row]] for row, idx_row in zip(indices, sorted_idx)])
    dists = np.concatenate([[row[idx_row]] for row, idx_row in zip(dists, sorted_idx)])
    return indices, dists

def make_similars_examples(embeds, labels, n_nearby, savename):
    """
    Runs through each sentence in labels and finds the n_nearby closest sentences, writing each
    tuple to a file that can be used as a set of examples for training. Each line of the file
    contains a tuple of similar sentences separated by: '|SEP|'
    """
    n_nearby += 1 # The nearest sentence for each example should be itself
    filename = '{}.similars'.format(savename)
    print 'Writing examples to file: {}'.format(filename)
    with open(filename, 'w') as out_file:
        batch_size = 25000
        num_processed = 0
        for batch in range(len(labels)/batch_size):
            query_sentences = labels[batch*batch_size:batch_size + batch*batch_size]
            query_vectors = embeds[batch*batch_size:batch_size + batch*batch_size]
            indices, _ = _find_nearest(query_vectors, embeds, n_nearby, batch_size=batch_size)
            for query_num in range(indices.shape[0]):
                # write each example (the nearest sentence should be the query itself)
                similar_sentences = []
                index_row = indices[query_num, :]
                for idx in index_row:
                    similar_sentences.append(labels[idx].strip())
                out_file.write(' |SEP| '.join(similar_sentences) + '\n')
            num_processed += batch_size
            print '{} examples written'.format(num_processed)
            sys.stdout.flush()

def show_similar(embeds, labels, n_examples=10, n_nearby=6):
    # Gather a random set of queries (sentences embeddings that we'll compare against)
    query_label_idx = random.randint(0, len(labels) - n_examples)
    query_vectors = embeds[query_label_idx:query_label_idx+n_examples, :]
    # Find indices for the embeddings that are nearest to the queries
    t = time.time()
    indices, dists = _find_nearest(query_vectors, embeds, n_nearby, batch_size=1000000)
    print indices.shape, dists.shape
    t = time.time() - t
    query_sentences = labels[query_label_idx: query_label_idx + dists.shape[0]]
    for query_num, query_sentence in enumerate(query_sentences):
        print "*******************************************************************"
        print query_sentence
        dist_row = dists[query_num, :]
        index_row = indices[query_num, :]
        for dist, idx in zip(dist_row, index_row):
            print dist, labels[idx]
    print 'Took {} seconds ({} s/per query)'.format(t, t/n_examples)
    return

    # TODO: gensim is much faster but doesn't handle duplicate entries very well, and goes OOM
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
    embeds = compute_sentence_embeds(args.labels, args.restore, args.savename, args.max_num,
        args.batch_size)
    labels = open(args.labels).readlines()[:embeds.shape[0]]
    show_similar(embeds, labels)
    if args.write_examples:
        make_similars_examples(embeds, labels, n_nearby=4, savename=args.savename)
    if args.project:
        projected_embeds = words.project_embeds(args.savename, embeds)
        words.plot_projections(projected_embeds, labels, args.savename)

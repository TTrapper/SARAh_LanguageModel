import os
import pickle
from collections import Counter
import random

import tensorflow as tf

class Data(object):
    def __init__(self, datadir, batch_size, max_word_len, max_line_len, go_stop_token=chr(0),
        unk_token=chr(1)):
        self.datadir = datadir
        self.go_stop_token = go_stop_token
        self.unk_token = unk_token
        (self.chr_to_freq,
         self.id_to_chr,
         self.chr_to_id) = create_chr_dicts(self.datadir, go_stop_token, unk_token,
                                            max_num_chrs=None)
        # Build pipeline for training and eval
        self.iterator, self.filepattern = make_pipeline(batch_size, self.chr_to_id,
            cycle_length=len(os.listdir(self.datadir)))
        src, trg = self.iterator.get_next()
        # Build pipeline for running inference
        self.src_place, self.trg_place, src, trg = make_inference_pipeline()

    def initialize(self, sess, filepattern):
        sess.run(self.iterator.initializer, feed_dict={self.filepattern:filepattern})

    def array_to_strings(self, array_3):
        return [' '.join([''.join([self.id_to_chr[c] for c in word if c != -1])
            for word in sentence]) for sentence in array_3]

def make_pipeline(batch_size, chr_to_id, cycle_length=128, shuffle_buffer=4096):
    do_shuffle = shuffle_buffer > 1
    sloppy = do_shuffle
    filepattern = tf.placeholder(dtype=tf.string, name='filepattern')
    numfiles = tf.placeholder(dtype=tf.int64, name='numfiles')
    filepaths = tf.data.Dataset.list_files(filepattern, shuffle=do_shuffle).repeat()
    file_processor = _make_file_processor_fn(chr_to_id)
    pair = filepaths.apply(tf.contrib.data.parallel_interleave(
        file_processor, sloppy=sloppy, cycle_length=cycle_length))
    pair = pair.shuffle(shuffle_buffer)
    pair = pair.batch(batch_size)
    pair = pair.prefetch(64)
    iterator = pair.make_initializable_iterator()
    return iterator, filepattern

def _make_file_processor_fn(chr_to_id):
    def examples_from_file(filename):
        lines = tf.data.TextLineDataset(filename)
        src = lines.map(_make_line_processor_fn(chr_to_id))
        trg = lines.skip(1)
        pair = tf.data.Dataset.zip((src, trg))
        return pair
    return examples_from_file

def _make_line_processor_fn(chr_to_id):
    def line_processor(line):
        line = _split_chars(line)
        line = chr_to_id.lookup(line)
        line = tf.sparse_tensor_to_dense(line)
        return line
    return line_processor

def _split_chars(line):
    line = tf.expand_dims(line, axis=0)
    line = tf.string_split(line, delimiter='')
    return line

def _make_pad_to_batch_fn(batch_size):
    def _pad_to_batch(*src_trg_batch):
	pad = lambda b: tf.SparseTensor(b.indices, b.values,
	    b.dense_shape + [batch_size - b.dense_shape[0], 0, 0])
	src = pad(src_trg_batch[0])
	trg = pad(src_trg_batch[1])
	return (src, trg)
    return _pad_to_batch

def make_inference_pipeline():
    src_place = tf.placeholder(dtype=tf.string, name='src_place')
    trg_place = tf.placeholder(dtype=tf.string, name='trg_place')
    src = process_line(src_place)
    src = tf.sparse.expand_dims(src, 0)
    trg = process_line(trg_place)
    trg = tf.sparse.expand_dims(trg, 0)
    return src_place, trg_place, src, trg

def create_chr_dicts(dirname, unk_token, max_num_chrs=None):
    chr_to_freq = None
    if chr_to_freq is None:
        chr_to_freq = Counter()
        for f in os.listdir(dirname):
            chr_to_freq += Counter(open(dirname + f, 'r').read().decode('utf-8'))
    if unk_token in chr_to_freq:
        raise ValueError('Reserved character found in dataset:\n' + str(chr_to_freq))
    chr_to_freq += Counter(unk_token)
    id_to_chr = chr_to_freq.most_common(max_num_chrs)
    id_to_chr = sorted([c[0] for c in id_to_chr])
    chr_to_id = tf.contrib.lookup.index_table_from_tensor(
        id_to_chr, default_value=id_to_chr.index(unk_token))
    return chr_to_freq, id_to_chr, chr_to_id

#### UTILITIES ####

# Trim a 2D sparse tensor representing a line of text to the dense_shape: [num_words, num_chars]
def sparse_trim(line, max_line_len, max_word_len):
    return tf.sparse_slice(line, [0, 0], [max_line_len, max_word_len])

# Get the length of each word from a 3D [batch, sentence, word] tensor of char IDs padded with -1
def get_word_lens(char_ids_3):
    mask_3 = tf.where(tf.equal(char_ids_3, -1), tf.zeros_like(char_ids_3),
        tf.ones_like(char_ids_3))
    wordlens_2 = tf.reduce_sum(mask_3, axis=2)
    return wordlens_2

# Get the length of each sentence from a 3D [batch, sentence, word] tensor of char IDs with -1 pads
def get_sentence_lens(char_ids_3, wordlens_2=None):
    wordlens_2 = get_word_lens(char_ids_3) if wordlens_2 is None else wordlens_2
    mask = tf.where(tf.equal(wordlens_2, 0), wordlens_2, tf.ones_like(wordlens_2))
    sentence_lens_1 = tf.reduce_sum(mask, axis=1)
    return sentence_lens_1

def char_array_to_txt(char_array_3):
    return '\n'.join([' '.join([''.join(word) for word in line]) for line in char_array_3])

def replace_pad_chrs(txt, replacements={chr(0):'_', chr(1):''}):
    for old, new in replacements.iteritems():
        txt = txt.replace(old, new)
    return txt

def getRandomSentence(paths, numSamples=1, sampleRange=1):
    randFile = random.choice(paths)
    lines = list(open(randFile))
    if len(lines) >= numSamples:
        samples = []
        for sampleNum in range(numSamples):
            idx = random.randint(0, len(lines)-sampleRange)
            samples.append(lines[idx:idx+sampleRange])
        return samples
    else:
        print 'WARNING: file has fewer lines than samples: ' + randFile
        return []


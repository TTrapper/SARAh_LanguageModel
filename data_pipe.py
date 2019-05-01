import os
import pickle
from collections import Counter
import random

import tensorflow as tf

class Data(object):
    def __init__(self, datadir, batch_size, max_word_len, max_line_len, go_stop_token=chr(0),
        eval_mode=False):
        """ """
        self.datadir = datadir
        self.go_stop_token = go_stop_token
        # Create chr to numerical id maps, assume utf-8 encoding
        self.id_to_chr = [chr(i) for i in range(256)]
        self.chr_to_id = tf.contrib.lookup.index_table_from_tensor(self.id_to_chr)
        # Build pipeline for training and eval
        self.iterator, self.filepattern = make_pipeline(batch_size, max_word_len, max_line_len,
            self.chr_to_id, cycle_length=1 if eval_mode else min(128, len(os.listdir(self.datadir))),
            shuffle_buffer= 1 if eval_mode else 4096, repeat=not eval_mode)
        (trg_vals, trg_row_lens) = self.iterator.get_next()
        (self.trg,
         self.trg_sentence_len,
         self.trg_word_len) = _compose_ragged_batch(trg_vals, trg_row_lens)
        # Build pipeline for running inferenc
        self.trg_place, trg = make_inference_pipeline(self.chr_to_id)
        (trg_vals, trg_row_lens) = trg
        (self.trg_inference, self.trg_sentence_len_inference, _) = _compose_ragged_batch(
            trg_vals, trg_row_lens)

    def initialize(self, sess, filepattern):
        sess.run(self.iterator.initializer, feed_dict={self.filepattern:filepattern})

    def array_to_strings(self, array_3, remove_stops=False):
        sentences = [' '.join([''.join([self.id_to_chr[c] for c in word if c != -1])
            for word in sentence]) for sentence in array_3]
        if remove_stops:
            sentences = [s.replace(self.go_stop_token, '') for s in sentences]
        return sentences



def make_pipeline(batch_size, max_word_len, max_line_len, chr_to_id, cycle_length=128,
    shuffle_buffer=4096, repeat=True):
    do_shuffle = shuffle_buffer > 1
    sloppy = do_shuffle
    filepattern = tf.placeholder(dtype=tf.string, name='filepattern')
    numfiles = tf.placeholder(dtype=tf.int64, name='numfiles')
    filepaths = tf.data.Dataset.list_files(filepattern, shuffle=do_shuffle)
    if repeat: # TODO try tf.data.experimental.shuffle_and_repeat(
        filepaths = filepaths.repeat()
    file_processor = _make_file_processor_fn(max_word_len, max_line_len)
    lines = filepaths.apply(tf.contrib.data.parallel_interleave(# TODO try tf.data.experimental.sample_from_datasets
        file_processor, sloppy=sloppy, cycle_length=cycle_length))
    lines = lines.shuffle(shuffle_buffer)
    lines = lines.batch(batch_size, drop_remainder=True) # NOTE: expands non-concat dim. Complicates sparse_batch_to_ragged
    lines = lines.map(lambda batch: chr_to_id.lookup(batch))
    lines = lines.map(lambda batch: _sparse_batch_to_ragged(batch, batch_size))
    lines = lines.prefetch(64) # TODO: try tf.data.experimental.prefetch_to_device
    iterator = lines.make_initializable_iterator()
    return iterator, filepattern

def _make_file_processor_fn(max_word_len, max_line_len):
    def examples_from_file(filename):
        lines = tf.data.TextLineDataset(filename)
        lines = lines.map(_process_line)
        lines = lines.map(lambda line: sparse_trim(line, max_line_len + 1, max_word_len + 1))
        return lines
    return examples_from_file

def _process_line(line):
    line = _split_words(line)
    line = _add_stop_words(line)
    line = _add_stop_chars(line)
    line = _split_chars(line)
    return line

def _split_words(line):
    line = tf.expand_dims(line, axis=0)
    return tf.string_split(line, delimiter=' ')

def _split_chars(line):
    def body(index, words):
        next_word = tf.sparse_slice(line, tf.to_int64(index), [1, 1]).values
        next_word = tf.string_split(next_word, delimiter='')
        words = tf.sparse_concat(axis=0, sp_inputs=[words, next_word], expand_nonconcat_dim=True)
        return index+[0, 1], words
    def condition(index, words):
        return tf.less(index[1], tf.size(line))
    i0 = tf.constant([0,1])
    firstWord = tf.string_split(tf.sparse_slice(line, [0,0], [1, 1]).values, delimiter='')
    _, line = tf.while_loop(condition, body, loop_vars=[i0, firstWord], back_prop=False)
    return line

def _add_stop_words(line):
    STOP = tf.SparseTensor(indices=[[0,0]], values=[chr(0)], dense_shape=[1,1])
    return tf.sparse_concat(axis=1, sp_inputs=[line, STOP])

def _add_stop_chars(line):
    values = tf.map_fn(lambda x:tf.string_join([x, chr(0)]), line.values, back_prop=False)
    line = tf.SparseTensor(line.indices, values, line.dense_shape)
    return line

def _sparse_batch_to_ragged(batch, batch_size):
    batch = tf.sparse.to_dense(batch, default_value=-1)
    sentence_lens_1 = get_sentence_lens(batch)
    sentence_lens_1 = tf.unstack(sentence_lens_1, num=batch_size, axis=0)
    batch = tf.unstack(batch, num=batch_size, axis=0)
    # Sentences are padded by tf.Dataset.map. Trim them to prevent empty rows in the RaggedTensor
    batch = [b[:l, :] for b,l in zip(batch, sentence_lens_1)]
    batch = [tf.RaggedTensor.from_tensor(b, padding=-1) for b in batch]
    batch = tf.stack(batch, axis=0)
    # Return RaggedTensor decomposed as vals and row lengths for compatibility with tf.Dataset.map
    return (batch.flat_values, batch.nested_row_lengths())

def _compose_ragged_batch(flat_values, nested_row_lengths):
    rag = tf.RaggedTensor.from_nested_row_lengths(flat_values, nested_row_lengths)
    rag_sentence_len = nested_row_lengths[0]
    rag_word_len = tf.RaggedTensor.from_row_lengths(nested_row_lengths[1], nested_row_lengths[0])
    return rag, rag_sentence_len, rag_word_len

def make_inference_pipeline(chr_to_id):
    trg_place = tf.placeholder(dtype=tf.string, name='trg_place')
    trg = _process_line(trg_place)
    trg = tf.sparse.expand_dims(trg, 0)
    trg = chr_to_id.lookup(trg)
    trg = _sparse_batch_to_ragged(trg, 1)
    return trg_place, trg

def create_chr_dicts(dirname, go_stop_token, unk_token, max_num_chrs=None):
    chr_to_freq = None
    if chr_to_freq is None:
        chr_to_freq = Counter()
        for f in os.listdir(dirname):
            chr_to_freq += Counter(open(dirname + f, 'r').read())
    if go_stop_token in chr_to_freq or unk_token in chr_to_freq:
        raise ValueError('Reserved character found in dataset:\n' + str(chr_to_freq))
    chr_to_freq += Counter(go_stop_token + unk_token)
    id_to_chr = chr_to_freq.most_common(max_num_chrs)
    id_to_chr = sorted([c[0] for c in id_to_chr])
    chr_to_id = tf.contrib.lookup.index_table_from_tensor(
        id_to_chr, default_value=id_to_chr.index(unk_token))
    return chr_to_freq, id_to_chr, chr_to_id

#### UTILITIES ####

def sparse_trim(line, max_line_len, max_word_len, trim_from_start=False):
    """
    line: a 2D sparse tensor with shape [num_words, num_chars] representence a line of text
    max_line_len: maximum number of words in the line to trim to
    max_word_len: maximum number of chars for each word to trim to
    trim_from_start: whether or not the words should be trimmed from the beginning (not chars)
    """
    if trim_from_start:
        num_words = line.dense_shape[0]
        start = tf.maximum(tf.constant(0, dtype=num_words.dtype), num_words - max_line_len)
        size = num_words - start
        return tf.sparse_slice(line, [start, 0], [size, max_word_len])
    else:
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

# Shuffles the words in a batch of sentences, respecting variable lengths (doesn't mix in padding)
def shuffle_words(char_ids_3, sentence_lens_1):
    max_len = tf.shape(char_ids_3)[1] # max(sentence_lens_1) breaks if all are smaller than this dim
    indices_2 = []
    for sentence_len in tf.unstack(sentence_lens_1):
        indices_1 = tf.random.shuffle(tf.range(sentence_len))
        pad_1 = tf.range(sentence_len, max_len)
        indices_2.append(tf.concat([indices_1, pad_1], axis=0))
    indices_2 = tf.stack(indices_2)
    return tf.batch_gather(char_ids_3, indices_2)

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


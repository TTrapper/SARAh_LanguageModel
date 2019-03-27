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
        self.iterator, self.filepattern = make_pipeline(batch_size, max_word_len, max_line_len,
            cycle_length=len(os.listdir(self.datadir)))
        src, trg = self.iterator.get_next()
        (self.src,
         self.trg,
         self.src_sentence_len,
         self.trg_sentence_len,
         self.trg_word_len) = sparse_chr_to_dense_id(self.chr_to_id, src, trg)
        # Build pipeline for running inference
        self.src_place, self.trg_place, src, trg = make_inference_pipeline()
        (self.src_inference,
         self.trg_inference,
         self.src_sentence_len_inference,
         self.trg_sentence_len_inference,
         _) = sparse_chr_to_dense_id(self.chr_to_id, src, trg)

    def initialize(self, sess, filepattern):
        sess.run(self.iterator.initializer, feed_dict={self.filepattern:filepattern})

    def array_to_strings(self, array_3):
        return [' '.join([''.join([self.id_to_chr[c] for c in word if c != -1])
            for word in sentence]) for sentence in array_3]

def make_pipeline(batch_size, max_word_len, max_line_len, cycle_length=128, shuffle_buffer=4096):
    do_shuffle = shuffle_buffer > 1
    sloppy = do_shuffle
    filepattern = tf.placeholder(dtype=tf.string, name='filepattern')
    numfiles = tf.placeholder(dtype=tf.int64, name='numfiles')
    filepaths = tf.data.Dataset.list_files(filepattern, shuffle=do_shuffle).repeat()
    file_processor = _make_file_processor_fn(max_word_len, max_line_len)
    pair = filepaths.apply(tf.contrib.data.parallel_interleave(
        file_processor, sloppy=sloppy, cycle_length=cycle_length))
    pair = pair.shuffle(shuffle_buffer)
    pair = pair.batch(batch_size)
    pair = pair.prefetch(64)
    iterator = pair.make_initializable_iterator()
    return iterator, filepattern

def _make_file_processor_fn(max_word_len, max_line_len):
    def examples_from_file(filename):
        lines = tf.data.TextLineDataset(filename)
        src = lines.map(_process_line)
        # TODO: src should be trimmed from the beginning, trg from the end
        src = src.map(lambda line: sparse_trim(line, max_line_len + 1, max_word_len + 1))
        trg = lines.skip(1)
        trg = trg.map(_process_line)
        trg = trg.map(lambda line: sparse_trim(line, max_line_len + 1, max_word_len + 1))
        pair = tf.data.Dataset.zip((src, trg))
        return pair
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
    src = _process_line(src_place)
    src = tf.sparse.expand_dims(src, 0)
    trg = _process_line(trg_place)
    trg = tf.sparse.expand_dims(trg, 0)
    return src_place, trg_place, src, trg

def create_chr_dicts(dirname, go_stop_token, unk_token, max_num_chrs=None):
    chr_to_freq = None
#    freq_path = basedir + '/chr_to_freq'
#    if os.path.exists(freq_path):
#        chr_to_freq = pickle.load(open(freq_path))
    if chr_to_freq is None:
        chr_to_freq = Counter()
        for f in os.listdir(dirname):
            chr_to_freq += Counter(open(dirname + f, 'r').read())
#        pickle.dump(chr_to_freq, open(freq_path, 'wb'))
    if go_stop_token in chr_to_freq or unk_token in chr_to_freq:
        raise ValueError('Reserved character found in dataset:\n' + str(chr_to_freq))
    chr_to_freq += Counter(go_stop_token + unk_token)
    id_to_chr = chr_to_freq.most_common(max_num_chrs)
    id_to_chr = sorted([c[0] for c in id_to_chr])
    chr_to_id = tf.contrib.lookup.index_table_from_tensor(
        id_to_chr, default_value=id_to_chr.index(unk_token))
    return chr_to_freq, id_to_chr, chr_to_id

def sparse_chr_to_dense_id(chr_to_id, src, trg):
    # Input to encoder
    src = chr_to_id.lookup(src)
    src = tf.sparse_tensor_to_dense(src, default_value=-1)
    # Decoder has different representations a different levels
    trg = chr_to_id.lookup(trg)
    trg = tf.sparse_tensor_to_dense(trg, default_value=-1)
    src_sentence_len = get_sentence_lens(src)
    trg_word_len = get_word_lens(trg)
    trg_sentence_len = get_sentence_lens(trg, trg_word_len)
    return src, trg, src_sentence_len, trg_sentence_len, trg_word_len

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


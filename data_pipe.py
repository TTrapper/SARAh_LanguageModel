import os
import pickle
from collections import Counter

import tensorflow as tf

class Data(object):
    def __init__(self, basedir, batch_size, max_word_len, max_line_len, go_stop_token=chr(0),
        unk_token=chr(1)):
        self.basedir = basedir
        self.traindir = basedir + 'train/'
        self.validdir = basedir + 'valid/'
        self.testdir = basedir + 'test/'
        self.go_stop_token = go_stop_token
        self.unk_token = unk_token
        self.iterator, self.filepattern = make_pipeline(batch_size, max_word_len, max_line_len,
            cycle_length=len(os.listdir(self.traindir)))
        self.chr_to_freq, self.id_to_chr = self.create_chr_dicts()
        self.chr_to_id = tf.contrib.lookup.index_table_from_tensor(
            self.id_to_chr, default_value=self.id_to_chr.index(self.unk_token))
        self.src, self.trg_word_enc, self.trg_word_dec, self.trg = self._prepare_data()
        self.src_sentence_len = self.get_sentence_lens(self.src)
        self.trg_word_len = self.get_word_lens(self.trg)
        self.trg_sentence_len = self.get_sentence_lens(self.trg_word_dec, self.trg_word_len)


    def create_chr_dicts(self, max_num_chrs=None):
        chr_to_freq = None
        freq_path = self.basedir + '/chr_to_freq'
        if os.path.exists(freq_path):
            chr_to_freq = pickle.load(open(freq_path))
        if chr_to_freq is None:
            chr_to_freq = Counter()
            for dirname in [self.traindir, self.validdir, self.testdir]:
                for f in os.listdir(dirname):
                    chr_to_freq += Counter(open(dirname + f, 'r').read())
            pickle.dump(chr_to_freq, open(freq_path, 'wb'))
        if self.go_stop_token in chr_to_freq or self.unk_token in chr_to_freq:
            raise ValueError('Reserved character found in dataset')
        id_to_chr = chr_to_freq.most_common(max_num_chrs)
        id_to_chr = sorted([c[0] for c in id_to_chr])
        id_to_chr.extend([self.go_stop_token, self.unk_token])
        return chr_to_freq, id_to_chr

    def _prepare_data(self):
        src, trg = self.iterator.get_next()
        # Input to encoder
        src = self.chr_to_id.lookup(src)
        src = tf.sparse_tensor_to_dense(src, default_value=-1)
        # Decoder has different representations a different levels
        trg = self.chr_to_id.lookup(trg)
        trg = tf.sparse_tensor_to_dense(trg, default_value=-1)
        # Input to word encoder of the target sentence decoder
        trg_word_enc = trg[:, :-1, 1:-1] # remove STOP word, GO/STOP chr
        # Input to word decoder of the target sentence Decoder
        trg_word_dec = trg[:, 1:, :-1] # remove GO word, STOP chr
        # Targets
        targets = trg[:, 1:, 1:] # remove GO word, GO chr
        return src, trg_word_enc, trg_word_dec, targets

    def get_word_lens(self, char_ids_3):
	mask_3 = tf.where(tf.equal(char_ids_3, -1), tf.zeros_like(char_ids_3),
	    tf.ones_like(char_ids_3))
	wordlens_2 = tf.reduce_sum(mask_3, axis=2)
	return wordlens_2

    def get_sentence_lens(self, char_ids_3, wordlens_2=None):
	wordlens_2 = self.get_word_lens(char_ids_3) if wordlens_2 is None else wordlens_2
	mask = tf.where(tf.equal(wordlens_2, 0), wordlens_2, tf.ones_like(wordlens_2))
	sentence_lens_1 = tf.reduce_sum(mask, axis=1)
	return sentence_lens_1




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
        src = lines.map(_split_words)
        src = src.map(_split_chars)
        src = src.map(lambda line: tf.sparse_slice(line, [0, 0], [max_line_len, max_word_len]))
        trg = lines.skip(1)
        trg = trg.map(_split_words)
        trg = trg.map(_add_go_stop_words)
        trg = trg.map(_add_go_stop_chars)
        trg = trg.map(_split_chars)
        # target lines and words are trimmed taking into account the added go/stop tokens.
        # note that the stop tokens will be trimmed from long sequences.
        trg = trg.map(lambda line: tf.sparse_slice(line, [0, 0],
            [max_line_len + 2, max_word_len + 2]))
        pair = tf.data.Dataset.zip((src, trg))
        return pair
    return examples_from_file

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

def _add_go_stop_words(line):
    GO = tf.SparseTensor(indices=[[0,0]], values=[chr(0)], dense_shape=[1,1])
    STOP = GO
    return tf.sparse_concat(axis=1, sp_inputs=[GO, line, STOP])

def _add_go_stop_chars(line):
    values = tf.map_fn(lambda x:tf.string_join([chr(0), x, chr(0)]), line.values, back_prop=False)
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


#### UTILITIES ####

def char_array_to_txt(char_array_3):
    return '\n'.join([' '.join([''.join(word) for word in line]) for line in char_array_3])

def replace_pad_chrs(txt, replacements={chr(0):'_', chr(1):''}):
    for old, new in replacements.iteritems():
        txt = txt.replace(old, new)
    return txt

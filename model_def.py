import tensorflow as tf

import layers
import data_pipe

class Model(object):
    def __init__(self, src_sentence_2, trg_sentence_2, num_chars, sos_token, config):
        self.config = config
        self.num_chars = num_chars
        # Encode source sentence
        src_sentence_3 = self.build_word_encoder(src_sentence_2)
        src_sentence_3_by_layer = self.build_sentence_encoder(src_sentence_3)
        # Add start of sequence tokens to offset the decoder's outputs from the labels by 1 'word'
        trg_sentence_2 = self.add_sos_word(trg_sentence_2, sos_token)
        # Generate target sentence word vectors by decoding source
        trg_sentence_3 = self.build_word_encoder(trg_sentence_2, reuse_vars=True)
        trg_words_3 = self.build_sentence_decoder(trg_sentence_3, src_sentence_3_by_layer)
        # Generate target sentence char predictions by decoding word vectors
        self.out_logits_3 = self.build_word_decoder(trg_words_3, trg_sentence_2)
        # Ops for generating predictions durng inference
        self.softmax_temp = tf.placeholder(self.out_logits_3.dtype, name='softmax_temp', shape=[])
        logits_2 = tf.reshape(self.out_logits_3, [-1, num_chars])
        predictions_2 = tf.multinomial(logits_2/self.softmax_temp, num_samples=1)
        self.predictions_3 = tf.reshape(predictions_2, tf.shape(self.out_logits_3)[:-1])

    def build_word_encoder(self, char_ids_2, reuse_vars=None):
        config = self.config
        with tf.variable_scope('word_encoder', reuse=reuse_vars):
            # Select char embeddings, contextualize them with RNN
            char_embeds_3 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_2)
            char_embeds_3, _ = layers.sarah_multilayer(char_embeds_3, None,
                config['word_encoder_layers'])
            # Gather every n'th contexualized char embed, these will act as implicit word embeds.
            chrs_per_word = config['chrs_per_word']
            indices_1 = tf.range(chrs_per_word - 1, tf.shape(char_ids_2)[1], delta=chrs_per_word)
            indices_2 = tf.tile(tf.expand_dims(indices_1, 0),
                tf.stack([tf.shape(char_ids_2)[0], 1]))
            word_vectors_3 = tf.batch_gather(char_embeds_3, indices_2)
        return word_vectors_3

    def add_sos_word(self, chr_ids_2, sos_token):
        # Create the sos 'word', which corresponds to a fixed length window of N chars.
        sos = tf.constant([self.config['chrs_per_word']*[sos_token]], dtype=chr_ids_2.dtype)
        sos = tf.tile(sos, tf.stack([tf.shape(chr_ids_2)[0], 1]))
        # The sos word is added to the beginning of the chr stream, and last N chrs are removed.
        chr_ids_2 = tf.concat([sos, chr_ids_2], axis=1)[:, :-self.config['chrs_per_word']]
        return chr_ids_2

    def build_sentence_encoder(self, word_vectors_3):
        with tf.variable_scope('sentence_encoder'):
            _, word_vectors_3_by_layer = layers.sarah_multilayer(word_vectors_3, seq_lens_1=None,
                layer_specs=self.config['sentence_encoder_layers'])
        return word_vectors_3_by_layer # num_layers * [batch, sentence_len, word_depth]

    def build_sentence_decoder(self, trg_sentence_3, src_3_by_layer):
        layer_specs = self.config['sentence_decoder_layers']
        # Add encoded src sentence to decocer spec, layer by layer
        for spec, src_3, in zip (layer_specs, src_3_by_layer):
            spec.update({'external_mem_3':src_3})
        with tf.variable_scope('sentence_decoder'):
            trg_sentence_3, _ = layers.sarah_multilayer(trg_sentence_3, seq_lens_1=None,
                layer_specs=layer_specs)
        return trg_sentence_3 # [batch, sentence_len, word_depth]

    def build_word_decoder(self, word_vectors_3, char_ids_2):
        config = self.config
#        gru = tf.keras.layers.GRU(256, return_sequences=True)
        with tf.variable_scope('word_encoder', reuse=True): # chr emsbeds shared with word encoder
            char_embeds_3 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_2)
        with tf.variable_scope('word_decoder'):
            # Project char embeds to word depth
            word_depth = word_vectors_3.shape.as_list()[-1]
            char_embeds_3 = layers.feed_forward(char_embeds_3, num_nodes=word_depth,
                activation_fn=layers.gelu, keep_prob=self.config['keep_prob'])
            # Tile word vectors so they are repeated for each associated chr
            word_vectors_2 = tf.reshape(word_vectors_3, [-1, word_depth])
            word_vectors_2 = tf.tile(word_vectors_2, [1, config['chrs_per_word']])
            word_vectors_3 = tf.reshape(word_vectors_2, [tf.shape(char_ids_2)[0], -1, word_depth])
            # Combine chr embeds with their associated word context
            char_embeds_3 += word_vectors_3
            char_embeds_3, _ = layers.sarah_multilayer(char_embeds_3, None,
                config['word_decoder_layers'])
        with tf.variable_scope('logits'):
            char_logits_3 = layers.feed_forward(char_embeds_3, num_nodes=self.num_chars)
        return char_logits_3

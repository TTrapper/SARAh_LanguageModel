import tensorflow as tf

import layers
import data_pipe

class Model(object):
    def __init__(self, src_sentence_3, src_sent_len_1, trg_word_enc_3, trg_sent_len_1, trg_word_dec_3,
        trg_word_len_2, num_chars, config):
        self.config = config
        self.num_chars = num_chars
        # Encode source sentence
        src_sentence_3 = self.build_word_encoder(src_sentence_3)
        src_sentence_3_by_layer = self.build_sentence_encoder(src_sentence_3, src_sent_len_1)
        # Generate target sentence word vectors by decoding source
        trg_word_enc_3 = self.build_word_encoder(trg_word_enc_3, reuse_vars=True)
        trg_word_enc_3 = self.build_sentence_decoder(trg_word_enc_3, trg_sent_len_1,
            src_sentence_3_by_layer, src_sent_len_1)
        # Generate target sentence chars by decoding word vectors
        self.out_logits_4 = self.build_word_decoder(trg_word_enc_3, trg_word_dec_3, trg_word_len_2)

    def create_spell_vector(self, char_embeds_4, pad=True):
        config = self.config
        max_sentence_len = tf.shape(char_embeds_4)[1]
        max_word_len = tf.shape(char_embeds_4)[2]
        unpadded_vector_size = config['char_embed_size'] * max_word_len
        spell_vectors_3 = tf.reshape(char_embeds_4, [-1, max_sentence_len, unpadded_vector_size])
        if pad:
            spell_vector_size = config['spell_vector_len'] * config['char_embed_size']
            padlen = spell_vector_size - unpadded_vector_size
            pad_tensor_3 = 0*tf.ones([tf.shape(spell_vectors_3)[0], max_sentence_len, padlen],
                dtype=tf.get_variable_scope().dtype)
            spell_vectors_3 = tf.concat([spell_vectors_3, pad_tensor_3], axis=2)
            # reshape is just a hack to define the inner dim, which is used to creat mlp vars later
            spell_vectors_3 = tf.reshape(spell_vectors_3, [-1,  max_sentence_len, spell_vector_size])
        return spell_vectors_3

    def build_word_encoder(self, char_ids_3, reuse_vars=None):
        config = self.config
        with tf.variable_scope('word_encoder', reuse=reuse_vars):
            # Select char embeddings, create fixed-len word-spelling representation
            char_embeds_4 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_3)
            spell_vectors_3 = self.create_spell_vector(char_embeds_4)
            # Send spelling-vectors through MLP
            word_vectors_3 = layers.mlp(spell_vectors_3, config['word_encoder_layers'])
        return word_vectors_3

    def build_sentence_encoder(self, word_vectors_3, sentence_lens_1):
        with tf.variable_scope('sentence_encoder'):
            _, word_vectors_3_by_layer = layers.sarah_multilayer(word_vectors_3, sentence_lens_1,
                self.config['sentence_encoder_layers'])
        return word_vectors_3_by_layer # num_layers * [batch, sentence_len, word_size]

    def build_sentence_decoder(self, trg_sentence_3, trg_sent_lens_1, src_3_by_layer,
        src_sent_lens_1):
        layer_specs = self.config['sentence_decoder_layers']
        for spec, src_3, in zip (layer_specs, src_3_by_layer):
            spec.update({'external_mem_3':src_3, 'external_seq_lens_1':src_sent_lens_1})
        with tf.variable_scope('sentence_decoder'):
            trg_sentence_3, _ = layers.sarah_multilayer(trg_sentence_3, trg_sent_lens_1,
                layer_specs)
        return trg_sentence_3 # [batch, sentence_len, word_size]

    def build_word_decoder(self, word_vectors_3, char_ids_3, word_lens_2):
        config = self.config
        # Loop function operates over 1 batch element, which corresponds to a sentence
        def word_rnn(args):
            char_embeds_3, word_lens_1, word_vectors_2 = args
            """
            char_embeds_3: [sentence_len, word_len, embed_size] char embeddings for this sentence
            word_lens_1: [sentence_len] length of each word 
            word_vectors_2: [sentence_len, word_vec_dim] outputs from sentence decoder
            """
            with tf.variable_scope('word_decoder_loop'):
                char_vectors_3 = layers.sarah(char_embeds_3, word_lens_1,
                    **config['word_decoder_sarah'])
                char_vectors_3 = layers.feed_forward(char_vectors_3,
                    num_nodes=word_vectors_2.shape.as_list()[-1], activation_fn=layers.gelu,
                    keep_prob=self.config['keep_prob'])
            word_vectors_3 = tf.expand_dims(word_vectors_2, axis=1) # prepare for broadcast 
            char_vectors_3 += word_vectors_3
            return char_vectors_3
        with tf.variable_scope('word_encoder', reuse=True):
            char_embeds_4 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_3)
        with tf.variable_scope('word_decoder'):
            char_vectors_4 = tf.map_fn(word_rnn, [char_embeds_4, word_lens_2, word_vectors_3],
                dtype=tf.float16)
            char_vectors_4 = layers.mlp(char_vectors_4, config['word_decoder_mlp'])
        with tf.variable_scope('logits'):
            char_logits_4 = layers.feed_forward(char_vectors_4, num_nodes=self.num_chars)
        return char_logits_4

import tensorflow as tf

import layers
import utils

class Model(object):
    def __init__(self, source_sentence_3, target_sentence_3, config):
        self.config = config
        # Encode source sentence
        src_word_vectors_3 = self.build_word_encoder(source_sentence_3)
        enc_word_vectors_3 = self.build_sentence_encoder(word_vectors_3, source_sentence_3)
        # Add GO word to target sentence, encode target words
        target_sentence_3 = utils.add_go(target_sentence_3, axis=1)
        trg_word_vectors_3 = self.build_word_encoder(target_sentence_3, reuse_vars=True)
        # Decode target sentence
        out_word_vectors_3 = self.build_sentence_decoder(
            trg_word_vectors_3, target_sentence_3, enc_word_vectors_3)
        # Remove GO word, add GO char to target sentence, decode target words to char logits
        target_sentence_3 = utils.add_stop(target_sentence_3[:, 1:, :], axis=1)
        target_sentence_3 = utils.add_go(target_sentence_3, axis=2)
        self.out_logits_4 = self.build_word_decoder(out_word_vectors_3, target_sentence_3)
        self.targets_3 = utils.add_stop(target_sentence_3[:, :, 1:], axis=2)

    def create_spell_vector(self, char_embeds_4, pad=True):
        config = self.config
        max_sentence_len_0 = tf.shape(char_embeds_4)[1]
        max_word_len_0 = tf.shape(char_embeds_4)[2]
        spell_vectors_3 = tf.reshape(char_embeds_4, [config['batch_size'],
            max_sentence_len_0, config['char_embed_size'] * max_word_len_0])
        if pad:
            embed_size = tf.shape(char_embeds_4)[3]
            spell_vector_size = config['spell_vector_len'] * embed_size
            spell_vectors_3 = utils.pad_or_trim(spell_vectors_3, size=spell_vector_size, axis=3)
        return spell_vectors_3

    def build_word_encoder(self, char_ids_3, reuse_vars=None):
        config = self.config
        with tf.variable_scope('word_encoder', reuse=reuse_vars):
            # Select char embeddings, create fixed-len word-spelling representation
            char_embeds_4 = layers.embedding(
                len(config['id2Char']), config['char_embed_size'], char_ids_3)
            spell_vectors_3 = self.create_spell_vector(char_embeds_4)
            # Send spelling-vectors through MLP
            word_vectors_3 = layers.mlp(spell_vectors_3, config['word_encoder_layer'])
            word_vectors_3 = tf.reshape(word_vectors_3,
                [config['batch_size'], max_sentence_len_0, tf.shape(word_vectors_2)[-1]])
        return word_vectors_3

    def build_sentence_encoder(self, word_vectors_3, char_ids_3):
        config = self.config
        with tf.variable_scope('sentence_encoder'):
            sentence_lens_1 = utils.jagged_array_lens(char_ids_3, axis=1)
            word_vectors_3 = layers.multi_sarah(word_vectors_3, sentence_lens_1,
                config['sentence_encoder_sarahs'))
        return word_vectors_3

    def build_sentence_decoder(self, word_vectors_3, char_ids_3, enc_word_vectors_3):
        config = self.config
        with tf.variable_scope('sentence_decoder'):
            sentence_lens_1 = utils.jagged_array_lens(char_ids_3, axis=1)
            word_vectors_3 = layers.multi_sarah(word_vectors_3, sentence_lens_1,
                config['sentence_decoder_sarahs'), condition=enc_word_vectors_3)
        return word_vectors_3

    def build_word_decoder(self, word_vectors_3, char_ids_3):
        config = self.config
        # Loop function operates over 1 batch element, which corresponds to a sentence
        def word_rnn(char_embeds_3, word_lens_1, word_vectors_2):
            """
            char_embeds_3: [sentence_len, word_len, embed_size] char embeddings for this sentence
            word_lens_1: [sentence_len] length of each word 
            word_vectors_2: [sentence_len, word_vec_dim] outputs from sentence decoder
            """
            char_vectors_3 = layers.sarah(char_embeds_3, word_lens_1, config['word_decoder_sarah'])
            char_vectors_3 = layers.mlp(char_vectors_3, [{'num_nodes': word_vectors_3.shape.as_list[-1],
                                                          'activation_fn': utils.gelu,
                                                          'keep_prob': config['keep_prob']}])
            word_vectors_3 = tf.expand_dims(word_vectors_2, axis=1) # prepare for broadcast 
            char_vectors_3 += word_vectors_3
            return char_vectors_3
        with tf.variable_scope('word_encoder', reuse=True):
            char_embeds_4 = layers.embedding(
                len(config['id2Char']), config['char_embed_size'], char_ids_3)
        with tf.variable_scope('word_decoder'):
            word_lens_2 = utils.jagged_array_lens(char_ids_3, axis=2)
            char_vectors_4 = tf.map_fn(word_rnn, [char_embeds_4, word_lens_2, word_vectors_3],
                dtype=tf.float16)
            char_logits_4 = layers.mlp(char_vectors_4, config['word_decoder_mlp'])
        return char_logits_4

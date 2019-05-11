import tensorflow as tf

import layers
import data_pipe

class Model(object):
    def __init__(self, sentences_3, sentence_lens_1, num_chars, config, inference_mode=False):
        sentences_3 = sentences_3.to_tensor(-1)
        self.config = config
        self.num_chars = num_chars
        self.inference_mode = inference_mode
        # Encode words and
        sentences_encoded_3 = self.add_go(sentences_3, axis=1)
        sentences_encoded_3 = self.build_word_encoder(sentences_encoded_3)
        self.word_embeds_3 = sentences_encoded_3
        sentences_encoded_3 = self.build_sentence_encoder(sentences_encoded_3, sentence_lens_1,
            config['sentence_encoder_layers'])
        # Generate target sentence char predictions by decoding word vectors
        self.out_logits_4 = self.build_word_decoder(sentences_encoded_3, sentences_3)
        # Ops for generating predictions durng inference
        self.softmax_temp = tf.placeholder(self.out_logits_4.dtype, name='softmax_temp', shape=[])
        logits_2 = tf.reshape(self.out_logits_4, [-1, num_chars])
        predictions_2 = tf.multinomial(logits_2/self.softmax_temp, num_samples=1)
        self.predictions_3 = tf.reshape(predictions_2, tf.shape(self.out_logits_4)[:-1])

    def add_go(self, char_ids_3, axis):
        assert axis == 1 or axis == 2
        if axis == 1: # GO word
            # Pad from a single char to keep consistent regargless of max word lens in the batch
            padding = -1*tf.ones_like(char_ids_3[:, 0, :-1])
            go = tf.zeros_like(char_ids_3[:, 0, 0])
            go = tf.concat([tf.expand_dims(go, axis=1), padding], axis=1)
        else : # GO char
            go = tf.zeros_like(char_ids_3[:, :, 0]) # extends beyond sentence_lens but that OK
        char_ids_3 = tf.concat([tf.expand_dims(go, axis=axis), char_ids_3], axis=axis)
        return char_ids_3[:, :-1, :] if axis == 1 else char_ids_3[:, :, :-1]

    def create_spell_vector(self, char_embeds_4, spell_vector_len, pad=True):
        config = self.config
        max_sentence_len = tf.shape(char_embeds_4)[1]
        max_word_len = tf.shape(char_embeds_4)[2]
        unpadded_vector_size = config['char_embed_size'] * max_word_len
        spell_vectors_3 = tf.reshape(char_embeds_4, [-1, max_sentence_len, unpadded_vector_size])
        if pad:
            spell_vector_size = spell_vector_len * config['char_embed_size']
            padlen = spell_vector_size - unpadded_vector_size
            pad_tensor_3 = 0*tf.ones([tf.shape(spell_vectors_3)[0], max_sentence_len, padlen],
                dtype=tf.get_variable_scope().dtype)
            spell_vectors_3 = tf.concat([spell_vectors_3, pad_tensor_3], axis=2)
            # reshape is just a hack to define the inner dim, which is used to creat mlp vars later
            spell_vectors_3 = tf.reshape(spell_vectors_3, [-1,  max_sentence_len, spell_vector_size])
        return spell_vectors_3

    def build_positional_char_embeds(self, char_ids_3, char_embed_size, mlp_layer_specs,
        word_len_limit):
        """ """
        char_ids_3 = char_ids_3[:, :, :word_len_limit] # potentially trim long words
        batch_size, max_sentence_len, max_word_len = tf.unstack(tf.shape(char_ids_3))
        # Select char embeddings
        with tf.variable_scope('chars'):
            char_embeds_4 = layers.embedding(self.num_chars, char_embed_size, char_ids_3)
        # Create char position ids for every possible char position in the batch (including padding)
        position_ids_1 = tf.range(max_word_len)
        position_ids_3 = tf.expand_dims(tf.expand_dims(position_ids_1, 0), 0)
        position_ids_3 = tf.tile(position_ids_3, [batch_size, max_sentence_len, 1])
        # Mask position_ids for padding chars
        where = tf.equal(char_ids_3, -1)
        position_ids_3 = tf.where(where, char_ids_3, tf.cast(position_ids_3, char_ids_3.dtype))
        # Convert position_ids to relative position (scalar between 0 and 1)
        word_lengths_3 = tf.reduce_max(position_ids_3, axis=2, keep_dims=True)
        word_lengths_3 = tf.where(tf.equal(word_lengths_3, 0), tf.ones_like(word_lengths_3), word_lengths_3)
        word_lengths_3 = tf.cast(word_lengths_3, char_embeds_4.dtype)
        position_ids_3 = tf.cast(position_ids_3, char_embeds_4.dtype)
        relative_positions_3 = position_ids_3 / word_lengths_3
        # Mask relative_positions for padding chars
        relative_positions_3 = tf.where(where, tf.zeros_like(relative_positions_3), relative_positions_3)
        # Combine char embeddings with their respective positions
        relative_positions_4 = tf.expand_dims(relative_positions_3, axis=3)
        positional_char_embeds_4 = tf.concat([char_embeds_4, relative_positions_4], axis=3)
        positional_char_embeds_4 = layers.mlp(positional_char_embeds_4, mlp_layer_specs)
        return positional_char_embeds_4

    def build_word_encoder(self, char_ids_3, reuse_vars=None):
        config = self.config
        with tf.variable_scope('char_encoder', reuse=reuse_vars):
            char_embeds_4 = self.build_positional_char_embeds(char_ids_3, config['char_embed_size'],
                config['char_encoder_mlp'], config['max_word_len'])
        with tf.variable_scope('word_encoder', reuse=reuse_vars):
            # Sum positional_char_embeds to get a word_vector, normalize and noise it.
            word_vectors_3 = layers.do_layer_norm(tf.reduce_sum(char_embeds_4, axis=2))
            shape_1 = tf.shape(word_vectors_3)
            word_vectors_2 = tf.reshape(word_vectors_3, [-1, shape_1[-1]])
            word_vectors_2 += layers.gaussian_noise(word_vectors_2, self.config['noise_level'])
            word_vectors_3 = tf.reshape(word_vectors_2, shape_1)
            # Pass word_vectors through an MLP
            word_vectors_3 = layers.mlp(word_vectors_3, config['word_encoder_mlp'])
        return word_vectors_3

    def build_sentence_encoder(self, word_vectors_3, sentence_lens_1, layer_specs):
        with tf.variable_scope('sentence_encoder'):
            # Build encoder
            encoded_3 = layers.sarah(word_vectors_3, sentence_lens_1, False, layer_specs)
        encoded_3 = self._checkpoint_encodings(encoded_3, sentence_lens_1, self.inference_mode)
        return encoded_3 # [batch, sentence_len, word_size]

    def _checkpoint_encodings(self, sentences_encoded_3, sentence_lens_1, inference_mode):
        """
        Sets up a checkpoint for inference mode on the context encoded word_vectors.
        Also creates a a fixed-length sentence representation from the mean of the word_vectors.
        """
        sentence_lens_2 = tf.expand_dims(sentence_lens_1, axis=1)
        sentence_lens_2 = tf.cast(sentence_lens_2, sentences_encoded_3.dtype)
        self.sentence_embeds_2 = tf.reduce_sum(sentences_encoded_3, axis=1)/sentence_lens_2
        self.sentences_encoded_checkpoint_3 = sentences_encoded_3
        if inference_mode:
            self.sentences_encoded_placeholder_3 = tf.placeholder(name='sentences_encoded_3',
                dtype=sentences_encoded_3.dtype, shape=sentences_encoded_3.shape)
            sentences_encoded_3 = self.sentences_encoded_placeholder_3
        return sentences_encoded_3

    def build_word_decoder(self, word_vectors_3, char_ids_3):
        config = self.config
        with tf.variable_scope('word_condition_projection'):
            word_vectors_3 = layers.mlp(word_vectors_3, self.config['sentence_decoder_projection'])
        with tf.variable_scope('word_decoder'):
            spell_vector_len = config['spell_vector_len']
            spell_vector_size = spell_vector_len * config['char_embed_size']
            spell_vector_size *= 2 # TODO make this factor configurable
            # Grab char embeds and concat them to spelling vector representations of words
            char_ids_3 = self.add_go(char_ids_3, axis=2)
            char_embeds_4 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_3)
            spell_vectors_3 = self.create_spell_vector(char_embeds_4, spell_vector_len)
            # Pass spelling vector through a layer that can see previous chars, but can't see ahead
            with tf.variable_scope('future_masked_spelling'):
                spell_vectors_projected_3 = layers.feed_forward(spell_vectors_3,
                    num_nodes=spell_vector_size, seq_len_for_future_mask=spell_vector_len)
            # Reshape word representation into individual char representations
            batch_size, sentence_len, word_len = tf.unstack(tf.shape(char_ids_3))
            char_size = spell_vectors_projected_3.shape.as_list()[-1]/spell_vector_len
            char_vectors_4 = tf.reshape(spell_vectors_projected_3,
                [batch_size, sentence_len, spell_vector_len, char_size])
            char_vectors_4 = char_vectors_4[:, :, :word_len, :]
            # Project each char_vector up to the size of the conditioning word_vector
            with tf.variable_scope('char_projection'):
                word_depth = word_vectors_3.shape.as_list()[-1]
                char_vectors_4 = layers.feed_forward(char_vectors_4, num_nodes=word_depth)
            # Add the conditioning word_vector to each char and pass result through an mlp
            char_vectors_4 += tf.expand_dims(word_vectors_3, axis=2)
            char_vectors_4 = layers.mlp(char_vectors_4, config['word_decoder_mlp'])
        with tf.variable_scope('logits'):
            char_logits_4 = layers.feed_forward(char_vectors_4, num_nodes=self.num_chars,
                noise_level=config['noise_level'])
        return char_logits_4

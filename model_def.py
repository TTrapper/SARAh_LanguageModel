import tensorflow as tf

import layers
import data_pipe

class Model(object):
    def __init__(self, src_sentence_3, src_sent_len_1, trg_sentence_3, trg_sent_len_1, num_chars,
        config, inference_mode=False):
        src_sentence_3 = src_sentence_3.to_tensor(-1)
        trg_sentence_3 = trg_sentence_3.to_tensor(-1)
        self.config = config
        self.num_chars = num_chars
        # Embed src words
        # Shuffle to prevent word order from being infered when they drop off the SARAh's mem array
        src_sentence_3 = data_pipe.shuffle_words(src_sentence_3, src_sent_len_1)
        src_word_embeds_3 = self.build_word_encoder(src_sentence_3)
        if inference_mode:
            self.src_word_embeds_3 = src_word_embeds_3
            self.src_word_placeholder_3 = tf.placeholder(name='src_word_embeds_3',
                dtype=self.src_word_embeds_3.dtype, shape=src_word_embeds_3.shape)
            src_word_embeds_3 = self.src_word_placeholder_3
        # Encode target sentence, conditioned on source words
        trg_sentence_encoded_3 = self.add_go(trg_sentence_3, axis=1)
        trg_sentence_encoded_3 = self.build_word_encoder(trg_sentence_encoded_3, reuse_vars=True)
        trg_sentence_encoded_3 = self.build_sentence_encoder(trg_sentence_encoded_3, trg_sent_len_1,
            src_word_embeds_3, src_sent_len_1, 'project_contextualized_words',
            config['sentence_encoder_layers'], reuse=False)
        # Generate target sentence char predictions by decoding word vectors
        self.out_logits_4 = self.build_word_decoder(trg_sentence_encoded_3, trg_sentence_3)
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

    def build_word_encoder(self, char_ids_3, reuse_vars=None):
        config = self.config
        with tf.variable_scope('word_encoder', reuse=reuse_vars):
            # Select char embeddings, create fixed-len word-spelling representation
            char_embeds_4 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_3)
            spell_vectors_3 = self.create_spell_vector(char_embeds_4, config['spell_vector_len'])
            with tf.variable_scope('project_spell_vec'):
                project_size = config['word_encoder_mlp'][-1]['num_nodes']
                spell_vectors_3 = layers.feed_forward(spell_vectors_3, project_size)
            # Send spelling-vectors through MLP
            word_vectors_3 = layers.mlp(spell_vectors_3, config['word_encoder_mlp'])
        return word_vectors_3

    def build_sentence_encoder(self, word_vectors_3, sentence_lens_1, condition_vectors_3,
        condition_seq_lens_1, condition_projection_scope, layer_specs, reuse=False):
        with tf.variable_scope('sentence_encoder', reuse=reuse):
            # Project the conditioning sequence to the depth of the layer cell. Assuming all layers
            # are  the same size, this results in one projection instead of one per layer.
            with tf.variable_scope(condition_projection_scope, reuse=tf.AUTO_REUSE):
                mem_size = layer_specs[0]['val_size'] + layer_specs[0]['key_size']
                condition_vectors_3 = layers.feed_forward(condition_vectors_3, mem_size)
                initial_state = (tf.expand_dims(condition_seq_lens_1, axis=1), condition_vectors_3)
            # Build encoder
            encoded_3 = layers.sarah(word_vectors_3, sentence_lens_1, False, layer_specs,
                initial_state)
        return encoded_3 # [batch, sentence_len, word_size]

    def build_word_decoder(self, word_vectors_3, char_ids_3):
        config = self.config
        with tf.variable_scope('word_condition_projection'):
            word_vectors_3 = layers.mlp(word_vectors_3, self.config['sentence_decoder_projection'])
        with tf.variable_scope('word_decoder'):
            spell_vector_len = config['spell_vector_len']
            # Grab char embeds and concat them to spelling vector representations of words
            char_ids_3 = self.add_go(char_ids_3, axis=2)
            char_embeds_4 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_3)
            spell_vectors_3 = self.create_spell_vector(char_embeds_4, spell_vector_len)
            # Create a weight_mask that prevents a dense layer from seeing future chars
            spell_vector_size = spell_vectors_3.shape.as_list()[-1]
            word_depth = word_vectors_3.shape.as_list()[-1]
            # Combine spelling vector with conditioning vector using future masked mlp
            spell_vectors_projected_3 = layers.feed_forward(spell_vectors_3, num_nodes=word_depth,
                activation_fn=layers.gelu, keep_prob=config['keep_prob'], layer_norm=True,
                seq_len_for_future_mask=spell_vector_len)
            word_and_chars_3 = word_vectors_3 + spell_vectors_projected_3
            word_vectors_3 = layers.mlp(word_and_chars_3, config['word_decoder_mlp'])
            # Reshape word representation into individual char representations
            batch_size, sentence_len, word_len = tf.unstack(tf.shape(char_ids_3))
            char_size = word_vectors_3.shape.as_list()[-1]/spell_vector_len
            char_vectors_4 = tf.reshape(word_vectors_3, [batch_size, sentence_len, spell_vector_len,
                char_size])
            char_vectors_4 = char_vectors_4[:, :, :word_len, :]
        with tf.variable_scope('logits'):
            char_logits_4 = layers.feed_forward(char_vectors_4, num_nodes=self.num_chars)
        return char_logits_4

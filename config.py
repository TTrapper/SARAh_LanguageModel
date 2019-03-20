import tensorflow as tf
import layers

def generate_config(keep_prob=0.9):
    char_embed_size = 32
    spell_vector_len = 20

    word_encoder_mlp = 8*[
       {'num_nodes':512,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob}]

    sentence_encoder_layers = 8*[
       {'val_size':512,
        'key_size':128,
        'num_heads':8,
        'keep_prob':keep_prob,
        'activation_fn':layers.gelu,
        'bidirectional':False}]

    # Decoder is same as encoder, but unidirectional
    sentence_decoder_layers = [l.copy() for l in sentence_encoder_layers]
    for layer in sentence_decoder_layers:
        layer['bidirectional'] = False

    word_decoder_size = 2*spell_vector_len*char_embed_size
    # project out word embeddings to same size as word_decoder layers
    sentence_decoder_projection = 1*[
       {'num_nodes':word_decoder_size,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob}]

    word_decoder_mlp = 8*[
       {'num_nodes':word_decoder_size,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob,
        'seq_len_for_future_mask':spell_vector_len}]

    config = {
              'max_grad_norm':1,
              'learn_rate':1e-4,
              'batch_size':32,
              'char_embed_size':char_embed_size,
              'spell_vector_len':spell_vector_len,
              'max_word_len':19,
              'max_line_len':33,
              'keep_prob':keep_prob,
              'word_encoder_mlp':word_encoder_mlp,
              'sentence_encoder_layers':sentence_encoder_layers,
              'sentence_decoder_layers':sentence_decoder_layers,
              'sentence_decoder_projection':sentence_decoder_projection,
              'word_decoder_mlp':word_decoder_mlp}
    return config

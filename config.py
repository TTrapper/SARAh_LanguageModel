import tensorflow as tf
import layers

def generate_config(keep_prob=0.9):
    word_encoder_mlp = 4*[
       {'num_nodes':512,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob}]

    sentence_encoder_layers = 2*[
       {'val_size':512,
        'key_size':64,
        'num_heads':2,
        'keep_prob':keep_prob,
        'activation_fn':layers.gelu,
        'bidirectional':True}]

    # Decoder is same as encoder, but unidirectional
    sentence_decoder_layers = [l.copy() for l in sentence_encoder_layers]
    for layer in sentence_decoder_layers:
        layer['bidirectional'] = False

    word_decoder_sarah = 2*[
       {'val_size':256,
        'key_size':32,
        'num_heads':2,
        'keep_prob':keep_prob,
        'activation_fn':layers.gelu,
        'bidirectional':False}]

    word_decoder_mlp = 4*[
       {'num_nodes':512,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob}]

    config = {
              'max_grad_norm':1,
              'learn_rate':1e-4,
              'batch_size':32,
              'char_embed_size':32,
              'spell_vector_len':20,
              'max_word_len':19,
              'max_line_len':33,
              'keep_prob':keep_prob,
              'word_encoder_mlp':word_encoder_mlp,
              'sentence_encoder_layers':sentence_encoder_layers,
              'sentence_decoder_layers':sentence_decoder_layers,
              'word_decoder_sarah':word_decoder_sarah,
              'word_decoder_mlp':word_decoder_mlp}
    return config

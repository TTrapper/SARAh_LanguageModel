import tensorflow as tf
import layers

def get_config(keep_prob=0.9):

    word_encoder_layers = 4*[
       {'val_size':512,
        'key_size':64,
        'num_heads':1,
        'keep_prob':keep_prob,
        'activation_fn':layers.gelu,
        'bidirectional':True}]

    sentence_encoder_layers = 4*[
       {'val_size':512,
        'key_size':64,
        'num_heads':1,
        'keep_prob':keep_prob,
        'activation_fn':layers.gelu,
        'bidirectional':True}]

    # Decoder is same as encoder, but unidirectional
    sentence_decoder_layers = [l.copy() for l in sentence_encoder_layers]
    for layer in sentence_decoder_layers:
        layer['bidirectional'] = False

    word_decoder_layers = 4*[
       {'val_size':512,
        'key_size':64,
        'num_heads':1,
        'keep_prob':keep_prob,
        'activation_fn':layers.gelu,
        'bidirectional':False}]


    config = {
              'max_grad_norm':100000,
              'learn_rate':1e-4,
              'batch_size':16,
              'char_embed_size':32,
              'chrs_per_word':8,
              'keep_prob':keep_prob,
              'word_encoder_layers':word_encoder_layers,
              'sentence_encoder_layers':sentence_encoder_layers,
              'sentence_decoder_layers':sentence_decoder_layers,
              'word_decoder_layers':word_decoder_layers}
    return config

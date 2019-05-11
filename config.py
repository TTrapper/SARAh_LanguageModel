import tensorflow as tf
import layers

def generate_config(keep_prob=1.0, noise_level=0.0):
    char_embed_size = 32
    spell_vector_len = 20
    max_line_len = 64

    char_encoder_mlp = 1*[
       {'num_nodes': 512,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob,
        'noise_level':0.0}]

    word_encoder_mlp = 8*[
       {'num_nodes':512,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob,
        'noise_level':noise_level}]

    sentence_encoder_layers = 8*[
       {'val_size':512,
        'key_size':128,
        'num_heads':8,
        'keep_prob':keep_prob,
        'noise_level':noise_level,
        'activation_fn':layers.gelu,
        'attention_window':16}]

    # project out word embeddings to same size as word_decoder layers
    sentence_decoder_projection = 1*[
       {'num_nodes':256,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob,
        'noise_level':noise_level}]

    word_decoder_mlp = 8*[
       {'num_nodes':256,
        'activation_fn':layers.gelu,
        'layer_norm':True,
        'keep_prob':keep_prob,
        'noise_level':noise_level}]

    config = {
              'max_grad_norm':5,
              'learn_rate':1e-4,
              'batch_size':20,
              'char_embed_size':char_embed_size,
              'spell_vector_len':spell_vector_len,
              'max_word_len':19,
              'max_line_len':max_line_len,
              'keep_prob':keep_prob,
              'noise_level':noise_level,
              'char_encoder_mlp':char_encoder_mlp,
              'word_encoder_mlp':word_encoder_mlp,
              'sentence_encoder_layers':sentence_encoder_layers,
              'sentence_decoder_projection':sentence_decoder_projection,
              'word_decoder_mlp':word_decoder_mlp}
    return config

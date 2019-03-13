import math
import numpy as np
import tensorflow as tf

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def do_layer_norm(tensor):
    return tf.contrib.layers.layer_norm(tensor, center=True, scale=True, trainable=True,
        begin_norm_axis=-1, begin_params_axis=-1)

def embedding(num_embeddings, embed_size, indices):
    embeddings = tf.get_variable('embeddings', [num_embeddings, embed_size])
    # -1 indices are used for padding. Mask them (and results) to avoid lookup complaints.
    mask = tf.where(tf.equal(indices, -1), tf.zeros_like(indices), tf.ones_like(indices))
    indices *= mask
    embeddings = tf.nn.embedding_lookup(embeddings, indices)
    mask = tf.cast(tf.expand_dims(mask, axis=mask.shape.ndims), embeddings.dtype)
    embeddings *= mask
    return embeddings

def feed_forward(inputs, num_nodes, activation_fn=None, layer_norm=True, keep_prob=1.0):
    assert keep_prob >= 0.0 and keep_prob <= 1.0
    with tf.variable_scope('feed_forward'):
        weights = tf.get_variable('weights', [inputs.shape.as_list()[-1], num_nodes])
        biases = tf.get_variable('biases', [num_nodes], initializer=tf.zeros_initializer())
    indepth = tf.shape(inputs)[-1]
    outshape = tf.concat([tf.shape(inputs)[:-1], [num_nodes]], axis=0)
    outputs = tf.matmul(tf.reshape(inputs, [-1, indepth]), weights) + biases
    if layer_norm:
        outputs = do_layer_norm(outputs)
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    if keep_prob < 1.0:
        outputs = tf.nn.dropout(outputs, keep_prob)
    return tf.reshape(outputs, outshape)

def mlp(inputs, layer_specs):
    """
    layer_specs: list of kwargs for feed_forward layers
    """
    for i, kwargs in enumerate(layer_specs):
        with tf.variable_scope('mlp_layer_%d' % i):
            outputs = feed_forward(inputs, **kwargs)
            inputs = outputs
    return outputs

def attention(values_3, keys_3, query_2, sequence_lengths_1):
    """
    batch_size, sequence_length, value_dim
    """
    batch_size, max_seq_len, key_size = tf.unstack(tf.shape(keys_3))
    queries_3 = tf.expand_dims(query_2, axis=1)
    # batched matmul gives dot product of the query against each of the keys
    weights_2 = tf.squeeze(tf.matmul(queries_3, keys_3, transpose_b=True), axis=1) # batch_size, seq_len
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        scaling = tf.get_variable('scaling', shape=(), dtype=weights_2.dtype)
    weights_2 = weights_2*scaling
    # mask the attention weights to the sequence lenghts before applying softmax
    sequence_mask_2 = tf.sequence_mask(sequence_lengths_1, maxlen=max_seq_len)
    mask_values_2 = -tf.ones_like(weights_2)*2**16 # -inf if dtype is float16
    weights_2 = tf.where(sequence_mask_2, weights_2, mask_values_2)
    # softmax and apply attention weights
    weights_2 = tf.nn.softmax(weights_2, axis=1)
    weights_3 = tf.expand_dims(weights_2, axis=2)
    values_3 *= weights_3
    values_2 = tf.reduce_sum(values_3, axis=1)
    return values_2

def slow_multihead_attention(values_3, keys_3, query_2, sequence_lengths_1, num_heads):
    # do attention for 1 head
    def map_fn(tensors):
        vals, keys, query = tensors
        attended = attention(vals, keys, query, sequence_lengths_1)
        attended = tf.expand_dims(attended, axis=1)
        return attended, keys, query
    batch_size, max_seq_len, key_size = tf.unstack(tf.shape(keys_3))
    val_size = tf.shape(values_3)[2]
    # split attention heads
    values_4 = tf.reshape(values_3, [batch_size, max_seq_len, num_heads, val_size/num_heads])
    values_4 = tf.transpose(values_4, [2, 0, 1, 3])
    keys_4 = tf.reshape(keys_3, [batch_size, max_seq_len, num_heads, key_size/num_heads])
    keys_4 = tf.transpose(keys_4, [2, 0, 1, 3])
    queries_3 = tf.reshape(query_2, [batch_size, num_heads, key_size/num_heads])
    queries_3 = tf.transpose(queries_3, [1, 0, 2])
    # run attention over each head
    attended_4, _, _ = tf.map_fn(map_fn, (values_4, keys_4, queries_3), parallel_iterations=num_heads)
    attended_3 = tf.squeeze(attended_4, axis=2)
    attended_3 = tf.transpose(attended_3, [1, 0 ,2])
    attended_2 = tf.reshape(attended_3, [batch_size, val_size])
    return attended_2

def _get_tensorshape(tensor):
    """
    Returns a list. Each item is an int if the size of that dim is known, and an tf.shape op if not.
    """
    shape = []
    for static_shape, shape_op in zip(tensor.shape.as_list(), tf.unstack(tf.shape(tensor))):
        shape.append(static_shape if static_shape is not None else shape_op)
    return shape

def multihead_attention(values_3, keys_3, query_2, sequence_lengths_1, num_heads=1):
    """
    values_3: batch_size, sequence_length, value_dim
    """
    batch_size, max_seq_len, key_size = _get_tensorshape(keys_3)
    # split attention heads
    key_size = key_size/num_heads
    keys_4 = tf.reshape(keys_3, [batch_size, max_seq_len, num_heads, key_size])
    keys_4 = tf.transpose(keys_4, [0, 2, 3, 1]) # batch_size, num_heads, key_size, seq_len
    queries_4 = tf.reshape(query_2, [batch_size, num_heads, 1, key_size])
    # batched matmul gives dot product of the query against each of the keys
    weights_3 = tf.squeeze(tf.matmul(queries_4, keys_4), axis=2) # batch_size, num_heads, seq_len
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        scaling = tf.get_variable('scaling', shape=(), dtype=weights_3.dtype)
    weights_3 = weights_3*scaling
    # mask the attention weights to the sequence lengths before applying softmax
    if sequence_lengths_1 is not None:
        sequence_mask_2 = tf.sequence_mask(sequence_lengths_1, maxlen=max_seq_len)
        sequence_mask_3 = tf.expand_dims(sequence_mask_2, axis=1)
        sequence_mask_3 = tf.tile(sequence_mask_3, [1, num_heads, 1])
        mask_values_3 = -tf.ones_like(weights_3)*2**16
        weights_3 = tf.where(sequence_mask_3, weights_3, mask_values_3)
    # softmax and apply attention weights
    weights_3 = tf.nn.softmax(weights_3, axis=2)
    weights_4 = tf.expand_dims(weights_3, axis=3)
    weights_4 = tf.transpose(weights_4, [0, 2, 1, 3]) # batch_size, seq_len, num_heads, 1;
    val_size = tf.shape(values_3)[2]
    values_4 = tf.reshape(values_3, [batch_size, max_seq_len, num_heads, val_size/num_heads])
    values_4 *= weights_4
    # recombine attention heads and reduce attended values
    values_3 = tf.reshape(values_4, [batch_size, max_seq_len, val_size])
    values_2 = tf.reduce_sum(values_3, axis=1)
    return values_2

def gru():
    # WIP
    raise NotImplemented()
    char_embeds_3 = layers.embedding(self.num_chars, config['char_embed_size'], char_ids_2)
    self.embeds = char_embeds_3
    # TODO: use Keras implemenation
    gru = tf.contrib.cudnn_rnn.CudnnGRU(len(self.config['word_decoder_layers']),
        self.config['word_decoder_layers'][0]['val_size'],
        input_mode='auto_select',
        direction='unidirectional',
        dropout=1 - self.config['keep_prob'], # NOTE > 0 dropout was causing an error
        dtype=tf.dtypes.float32)

    # NOTE cudnn expects time major ?? Keras version may not
    char_embeds_3 = tf.transpose(char_embeds_3, [1,0,2])
    char_embeds_3 = gru.apply(char_embeds_3)[0]
    self.grud = char_embeds_3
    char_embeds_3 = tf.transpose(char_embeds_3, [1,0,2])

def sarah(inputs_3, seq_lens_1, bidirectional, layer_specs):
    """
    inputs_3: Tensor with shape: [batch_size, max_seq_len, input_depth]
    seq_lens_1: Tensor with shape [batch_size]
    layer_specs: list containing kwargs for one or more SelfAttentiveCell
    bidirectional: boolean whether to create a backward_cell
    """
    cells = [SelfAttentiveCell(**kwargs) for kwargs in layer_specs]
    if bidirectional:
        backward_cells = [SelfAttentiveCell(**kwargs) for kwargs in layer_specs]
        outputs_3, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells,
            backward_cells, inputs_3, dtype=inputs_3.dtype, sequence_length=seq_lens_1)
        return outputs_3
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs_3, state = tf.nn.dynamic_rnn(cell, inputs=inputs_3, dtype=inputs_3.dtype,
            sequence_length=seq_lens_1)
        return outputs_3

class SelfAttentiveCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, val_size, key_size, num_heads=1, external_mem_array=None,
        external_seq_lens=None, keep_prob=1.0, activation_fn=gelu):
        """ """
        super(SelfAttentiveCell, self).__init__()
        self.attention_window = 32 #TODO make this configurable
        self.project_inputs = False
        self.val_size = val_size
        self.key_size = key_size
        self.mem_size = key_size + val_size
        self.num_keys = 1 if external_mem_array is None else 2
        self.num_heads = num_heads
        self.keep_prob = keep_prob
        self.external_mem_array = external_mem_array
        self.external_seq_lens = external_seq_lens
        self.activation_fn = activation_fn

    @property
    def state_size(self):
        return (1, self.mem_size * self.attention_window)

    @property
    def output_size(self):
        return self.val_size + self.num_keys*self.key_size

    def build(self, inputs_shape):
        input_depth = inputs_shape[1].value
        if input_depth is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        if input_depth != self.output_size:
            self._project_w = self.add_variable('project_w', shape=[input_depth, self.output_size])
            self._project_b = self.add_variable('project_b', shape=[self.output_size],
                initializer=tf.zeros_initializer(dtype=self.dtype))
            self.project_inputs = True
        if self.external_mem_array is not None:
            if self.external_mem_array.shape[-1] != self.mem_size:
                self.external_mem_array = feed_forward(self.external_mem_array, self.mem_size)
            self.external_vals = self.external_mem_array[:, :, :self.val_size]
            self.external_keys = self.external_mem_array[:, :, -self.key_size:]
        kernel_in = self.val_size
        kernel_out = self.val_size + self.num_keys*self.key_size
        self._kernel = self.add_variable('weights', shape=[kernel_in, kernel_out])
        self._bias = self.add_variable('biases', shape=[kernel_out],
            initializer=tf.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        if self.project_inputs:
            inputs = tf.matmul(inputs, self._project_w)
            inputs = tf.nn.bias_add(inputs, self._project_b)
        seq_lens, memory = state
        memory = tf.reshape(memory, [-1, self.attention_window, self.mem_size])
        query = inputs[:, self.val_size:self.val_size+self.key_size]
        value = inputs[:, :self.val_size]
        context = self._attend_to_memory(memory, query, seq_lens)
        if self.external_mem_array is not None:
            query = inputs[:, -self.key_size:] # not the same query used for internal mem
            context += self._attend_to_external(query)
        value = do_layer_norm(value + context)
        output = tf.matmul(value, self._kernel)
        output = tf.nn.bias_add(output, self._bias)
        output = do_layer_norm(output)
        # Extract new mem val and add it to memory. Values are added to the beggining to align with
        # sequence masking during attention. This means that memory is in reverse order.
        new_mem_val = tf.expand_dims(output[:, :self.mem_size], axis=1)
        memory = tf.concat([new_mem_val, memory[:, :-1, :]], axis=1)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        if self.keep_prob < 1.0:
            output = tf.nn.dropout(output, keep_prob=self.keep_prob)
        return output, (seq_lens + 1, tf.reshape(memory, [-1, self.attention_window*self.mem_size]))

    def _attend_to_memory(self, memory_3, query, seq_lens):
        seq_lens = tf.squeeze(seq_lens, axis=1)
        seq_lens = tf.math.minimum(seq_lens, self.attention_window)
        values = memory_3[:, :, :self.val_size]
        keys = memory_3[:, :, -self.key_size:]
        return multihead_attention(values, keys, query, seq_lens, self.num_heads)

    def _attend_to_external(self, query):
        return multihead_attention(self.external_vals, self.external_keys, query,
            self.external_seq_lens, self.num_heads)

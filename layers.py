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

def gaussian_noise(inputs_2, noise_level):
    """
    Produces a perturbation for each row of inputs_2 with a gaussian that is proportional to the
    row's variance. The perturbation has mean 0 and stddev equal to noise_level * stddev(row).
    inputs_2: a 2D tensor to which row-wise pertubations will be added.
    noise_level: a scalar multiplier for the perturbation's stddev.
    """
    _, variance_1 = tf.nn.moments(inputs_2, axes=[-1])
    stddev_2 = tf.expand_dims(noise_level*(variance_1**0.5), axis=1)
    pertubation_2 = stddev_2 * tf.random_normal(tf.shape(inputs_2))
    return pertubation_2

def feed_forward(inputs, num_nodes, activation_fn=None, layer_norm=True, keep_prob=1.0,
        seq_len_for_future_mask=None, noise_level=0):
    """
    seq_len_for_future_mask: Used to construct a weight mask that prevents the layer from looking
        into the future. Assumes the input is a concatenated sequence of items, such as char embeds.
    """
    assert keep_prob >= 0.0 and keep_prob <= 1.0
    with tf.variable_scope('feed_forward'):
        weights = tf.get_variable('weights', [inputs.shape.as_list()[-1], num_nodes])
        biases = tf.get_variable('biases', [num_nodes], initializer=tf.zeros_initializer())
    inshape = tf.shape(inputs)
    outshape = tf.concat([tf.shape(inputs)[:-1], [num_nodes]], axis=0)
    indepth = inputs.shape.as_list()[-1]
    if seq_len_for_future_mask is not None:
        weight_mask = future_mask(seq_len_for_future_mask, indepth, num_nodes)
        weights = tf.where(weight_mask, weights, tf.zeros_like(weights))
    inputs = tf.reshape(inputs, [-1, indepth])
    outputs = tf.matmul(inputs, weights) + biases
    if seq_len_for_future_mask is not None:
        # Flatten implied sequence to prevent layer_norm/noise from leaking future information
        outputs = tf.reshape(outputs, [-1, num_nodes/seq_len_for_future_mask])
    if layer_norm:
        outputs = do_layer_norm(outputs)
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    if noise_level > 0:
        outputs += gaussian_noise(outputs, noise_level)
    if keep_prob < 1.0:
        outputs = tf.nn.dropout(outputs, keep_prob)
    outputs = tf.reshape(outputs, outshape)
    # Always add residual connection if the shapes match up
    if indepth == num_nodes:
        outputs += tf.reshape(inputs, inshape)
    return outputs

def future_mask(seq_len, in_size, out_size):
    """
    Create a weight mask that prevents a dense layer's parameters from looking forward. This assumes
    that the input to the dense layer is a concatenated sequence of items, such as a list of char
    embeddings concatenated to represent a word. An MLP built from layers masked in this way can
    act as a sequence decoder.

    seq_len: the length of the concatenated sequence represented by the input tensor.
    in_size: the number input nodes (number of rows for the weights)
    out_size: the number output nodes (number of columns for the weights)
    """
    assert in_size % seq_len == 0 and out_size % seq_len == 0
    item_size = in_size / seq_len
    new_item_size = int(item_size * (float(out_size)/in_size))
    mask_lengths = range(out_size, 0, -new_item_size)
    mask_lengths = [item_size*[l] for l in mask_lengths] # adjust mask to cover embedding size
    mask_lengths = [item for sublist in mask_lengths for item in sublist] # flatten
    mask = tf.reverse(tf.sequence_mask(mask_lengths), axis=[1])
    return mask

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

def multihead_attention(values_3, keys_3, query_2, sequence_lengths_1, num_heads=1):
    """
    values_3: batch_size, sequence_length, value_dim
    """
    batch_size, max_seq_len, key_size = tf.unstack(tf.shape(keys_3))
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
    # mask the attention weights to the sequence lenghts before applying softmax
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

def sarah(inputs_3, seq_lens_1, bidirectional, layer_specs, initial_state=None):
    """
    inputs_3: Tensor with shape: [batch_size, max_seq_len, input_depth]
    seq_lens_1: Tensor with shape [batch_size]
    layer_specs: list containing kwargs for one or more SelfAttentiveCell
    bidirectional: boolean whether to create a backward_cell
    initial_state: tuple of tensors ([batch_size, 1], [batch_size, mem_length, mem_size]
                    The inital sequence lengths and contents of the SARAh's mem array
    """
    cells = [SelfAttentiveCell(**kwargs) for kwargs in layer_specs]
    if initial_state is not None:
        if bidirectional:
            raise NotImplemented('bidirectional SARAh does not yet support setting initial_state')
        mem_vals = initial_state[1]
        attention_window = layer_specs[0]['attention_window'] #FIXME assumes same size on all layers
        batch_size, mem_length, mem_size = tf.unstack(tf.shape(mem_vals))
        # pad mem_vals to fit the attention_window
        pad_len = attention_window - mem_length
        pad_tensor_3 = tf.expand_dims(tf.zeros_like(mem_vals[:, 0, :]), axis=1)
        pad_tensor_3 = tf.tile(pad_tensor_3, [1, pad_len, 1])
        mem_vals = tf.concat([mem_vals, pad_tensor_3], axis=1)
        # flatten the sequence of mem_vals to [batch_size, mem_length*mem_size]
        mem_vals = tf.reshape(mem_vals, [batch_size, attention_window*mem_size])
        initial_state = (initial_state[0], mem_vals)
        initial_state=tuple(len(cells)*[initial_state])
    if bidirectional:
        backward_cells = [SelfAttentiveCell(**kwargs) for kwargs in layer_specs]
        outputs_3, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells,
            backward_cells, inputs_3, dtype=inputs_3.dtype, sequence_length=seq_lens_1)
        return outputs_3
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        outputs_3, state = tf.nn.dynamic_rnn(cell, inputs=inputs_3, dtype=inputs_3.dtype,
            sequence_length=seq_lens_1, initial_state=initial_state)
        return outputs_3

class SelfAttentiveCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, val_size, key_size, num_heads=1, keep_prob=1.0, noise_level=0.0,
        activation_fn=None, attention_window=32):
        """ """
        self.attention_window = attention_window
        super(SelfAttentiveCell, self).__init__()
        self.val_size = val_size
        self.key_size = key_size
        self.mem_size = key_size + val_size
        self.num_heads = num_heads
        self.keep_prob = keep_prob
        self.noise_level = noise_level
        self.activation_fn = activation_fn
        self.project_inputs = False

    @property
    def state_size(self):
        return (1, self.mem_size * self.attention_window)

    @property
    def output_size(self):
        return self.val_size + 2*self.key_size

    def build(self, inputs_shape):
        input_depth = inputs_shape[1].value
        if input_depth is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        if input_depth != self.output_size:
            self.project_w = self.add_variable('project_w', shape=[input_depth, self.output_size])
            self.project_b = self.add_variable('project_b', shape=[self.output_size])
            self.project_inputs = True
        self._kernel = self.add_variable('weights', shape=[self.val_size, self.output_size])
        self._bias = self.add_variable('biases', shape=[self.output_size],
            initializer=tf.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        if self.project_inputs:
            inputs = tf.nn.bias_add(tf.matmul(inputs, self.project_w), self.project_b)
        seq_lens, memory = state
        # Split input into QKV, concat it to memory, and apply attention
        query = inputs[:, -self.key_size:]
        key_value = inputs[:, :self.mem_size]
        memory = tf.reshape(memory, [-1, self.attention_window, self.mem_size])
        memory = tf.concat([tf.expand_dims(key_value, axis=1), memory], axis=1)
        context = self._attend_to_memory(memory, query, seq_lens + 1)
        memory = memory[:, 1:-1, :] # Remove oldest mem_val and the concatenated input val
        # Run attended history through a feed forward layer to create a new mem_val/output
        output = tf.matmul(context, self._kernel)
        output = tf.nn.bias_add(output, self._bias)
        output = do_layer_norm(output)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        if self.noise_level > 0.0:
            output += gaussian_noise(output, self.noise_level)
        # Extract new mem val and add it to memory. Values are added to the beggining to align with
        # sequence masking during attention. This means that memory is in reverse order.
        new_mem_val = tf.expand_dims(output[:, :self.mem_size], axis=1)
        memory = tf.concat([new_mem_val, memory], axis=1)
        if self.keep_prob < 1.0:
            output = tf.nn.dropout(output, keep_prob=self.keep_prob)
        output += inputs # residual connection
        return output, (seq_lens + 1, tf.reshape(memory, [-1, self.attention_window*self.mem_size]))

    def _attend_to_memory(self, memory_3, query, seq_lens):
        seq_lens = tf.squeeze(seq_lens, axis=1)
        seq_lens = tf.math.minimum(seq_lens, self.attention_window)
        values = memory_3[:, :, :self.val_size]
        keys = memory_3[:, :, -self.key_size:]
        return multihead_attention(values, keys, query, seq_lens, self.num_heads)

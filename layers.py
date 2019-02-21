import math
import numpy as np
import tensorflow as tf

def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def do_layer_norm(tensor):
    # NOTE layer norm is disabled
    return tensor
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

def sarah_multilayer(inputs_3, seq_lens_1, layer_specs):
    """
    layer_specs: list of kwargs for each SARAh layer
    """
    outputs_by_layer = []
    for i, kwargs in enumerate(layer_specs):
        with tf.variable_scope('SARAh_layer_%d' % i):
            outputs_3 = sarah(inputs_3, seq_lens_1, **kwargs)
            outputs_by_layer.append(outputs_3)
            inputs_3 = outputs_3
    return outputs_3, outputs_by_layer

def sarah(inputs_3, seq_lens_1, val_size, key_size, num_heads, keep_prob=1.0, activation_fn=None,
        external_mem_3=None, external_seq_lens_1=None, bidirectional=False):
    """
    inputs_3: [batch_size, seq_len, dim]
    external_mem_3 = [batch_size, seq_len, val_size + key_size]
    """
    cell = SelfAttentiveCell(val_size, key_size, num_heads, external_mem_3,
        external_seq_lens_1)
    input_depth = inputs_3.shape.as_list()[-1]
    if input_depth != cell.output_size:
        with tf.variable_scope('sarah_in_projection'):
            inputs_3 = feed_forward(inputs_3, cell.output_size, layer_norm=True, keep_prob=keep_prob)
    if bidirectional:
        with tf.variable_scope('backward'):
            backward_cell = SelfAttentiveCell(val_size, key_size, num_heads, external_mem_3,
                external_seq_lens_1)
        outputs_3, _ = tf.nn.bidirectional_dynamic_rnn(cell, backward_cell, inputs_3, seq_lens_1, parallel_iterations=1,
            dtype=tf.get_variable_scope().dtype)
        outputs_3 = outputs_3[0] + outputs_3[1] # combine forward/backward passes
    else:
        outputs_3, _ = tf.nn.dynamic_rnn(cell, inputs_3, seq_lens_1, parallel_iterations=1,
            dtype=tf.get_variable_scope().dtype)
    outputs_3 = do_layer_norm(outputs_3)
    if activation_fn is not None:
        outputs_3 = activation_fn(outputs_3)
    return outputs_3

class SelfAttentiveCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, val_size, key_size, num_heads=1, external_mem_array=None,
        external_seq_lens=None, keep_prob=1.0):
        """ """
        super(SelfAttentiveCell, self).__init__()
        self.val_size = val_size
        self.key_size = key_size
        self.mem_size = key_size + val_size 
        self.num_keys = 1 if external_mem_array is None else 2
        self.num_heads = num_heads
        self.keep_prob = keep_prob
        self.memory = tf.TensorArray(tf.get_variable_scope().dtype, 0, dynamic_size=True,
            clear_after_read=False, element_shape=[None, self.val_size+self.key_size], name='memTA')
        self.external_mem_array = external_mem_array
        if external_mem_array is not None:
            if external_mem_array.shape[-1] != val_size + key_size:
                raise ValueError("External mem has shape %s but must have depth of internal mem: %s"
                    % (external_mem_array.shape, val_size + key_size))
            self.external_vals = external_mem_array[:, :, :val_size]
            self.external_keys = external_mem_array[:, :, -key_size:]
            self.external_seq_lens = external_seq_lens

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self.val_size + self.num_keys*self.key_size

    def build(self, inputs_shape):
        input_depth = inputs_shape[1].value
        if input_depth is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        if input_depth != self.output_size:
            raise ValueError("Input size is %s but it must be the same as size of state + keys: %s"
                % (input_depth, self.output_size))
        kernel_in = self.val_size
        kernel_out = self.val_size + self.num_keys*self.key_size
        self._kernel = self.add_variable('weights', shape=[kernel_in, kernel_out])
        self._bias = self.add_variable('biases', shape=[kernel_out],
            initializer=tf.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        # There is one time counter per batch element. If the cell is unrolled with sequence_length
        # parameter, the time counter will stop at the provided sequence_length for each sequence
        # in the batch. We take the max here to always write to a new index of the memory array.
        # FIXME: if the longest sequence length specified is shorter than the number of elements
        # in the input tensor, then this will fail since the time counter will get stuck at that
        # specified sequence_length. What is wrong with using self.memory.size() ?
        time = tf.to_int32(tf.reduce_max(state))
        # memory is empty for the first timestep
        attended = tf.cond(tf.equal(time, 0),
            lambda: tf.zeros_like(inputs[:, :self.val_size]),
            lambda: self._attend_to_memory(inputs, tf.squeeze(state)))
        inputs = attended + inputs[:, :self.val_size]
        if self.external_mem_array is not None:
            inputs += self._attend_to_memory(inputs, None, True)
        inputs = do_layer_norm(inputs)
        output = tf.matmul(inputs, self._kernel)
        output = tf.nn.bias_add(output, self._bias)
        if self.keep_prob < 1.0:
            output = tf.nn.dropout(output, keep_prob=self.keep_prob)
        # Add new value to memory
        self.memory = self.memory.write(time, output[:, :self.mem_size])
        # FIXME: this is a hack which for some reason is needed to get the above write to take
        output += 1e-15*self.memory.read(time)[0,0] # forces output to depend on TA read.
        return output, state + 1

    def _attend_to_memory(self, inputs, seq_lens, external=False):
        if external:
            # In this case the query is always the rightmost part of the inputs.
            query = inputs[:, -self.key_size:]
            return multihead_attention(self.external_vals, self.external_keys, query,
                self.external_seq_lens, self.num_heads)
        else:
            # The query follows after the value portion of inputs.
            query = inputs[:, self.val_size:self.val_size+self.key_size]
            memory_3 = self.memory.stack() # [seq, batch, depth]
            memory_3 = tf.transpose(memory_3, [1, 0, 2]) # [batch, seq, depth]
            values = memory_3[:, :, :self.val_size]
            keys = memory_3[:, :, -self.key_size:]
            return multihead_attention(values, keys, query, seq_lens, self.num_heads)

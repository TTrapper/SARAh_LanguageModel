import TFlibs.networkcrafter as nc
import tensorflow as tf

def attention(memoryTensor, query, batchSize, numStateNodes, numKeyNodes, numHeads, includeKeys=False):
    keys = memoryTensor[:, :, numStateNodes:]
    keys = tf.reshape(keys, [-1, batchSize, numHeads, numKeyNodes/numHeads])
    keys = tf.transpose(keys, perm=[1, 2, 3, 0]) # [batchSize, numHeads, numKeyNodes, seq]
    query = tf.reshape(query, [1, batchSize, numHeads, numKeyNodes/numHeads])
    query = tf.transpose(query, perm=[1, 2, 0, 3]) # [batchSize, numHeads, 1, numKeyNodes]

    # Each mem item is weighted by attention: the dot product of its key and the current query
    weights = tf.squeeze(tf.matmul(query, keys), axis=2) # [batchSize, numHeads, seq]
    weights = weights/tf.sqrt(tf.cast(numKeyNodes/numHeads, tf.float32))
    # ^^ TODO: is this scaled properly?
    weights = tf.where(tf.equal(weights, 0), -1024*tf.ones_like(weights), weights)
    # ^^ TODO: A weight of exactly 0 is assumed to be for an empty initial state or a padded seq.
    weights = tf.nn.softmax(weights, axis=2)
    weights = tf.transpose(weights, [2, 0, 1]) # [seq, batchSize, numHeads]
    weights = tf.expand_dims(weights, axis=3)
    # Attend values
    numAttendedNodes = numStateNodes if not includeKeys else numStateNodes+numKeyNodes
    weights = tf.tile(weights, [1, 1, 1, numAttendedNodes/numHeads])
    weights = tf.reshape(weights, [-1, batchSize, numAttendedNodes])
    attended = weights * memoryTensor[:, :, :numAttendedNodes]
    attended = tf.reduce_sum(attended, axis=0)
    attended = tf.reshape(attended, [batchSize, numAttendedNodes])
    return attended

class SelfAttentiveCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, numStateNodes, numKeyNodes, batchSize, numHeads=1, initialMemValues=None,
            extraMemArray=None, activations=tf.nn.relu, layerNorm=False):
        super(SelfAttentiveCell, self).__init__()
        self.layerNorm = layerNorm
        self.numStateNodes = numStateNodes*numHeads
        self.numKeyNodes = numKeyNodes*numHeads
        self.outputSize = (numStateNodes+numKeyNodes)*numHeads
        self.batchSize = batchSize
        self.numHeads = numHeads
        self.activations = activations
        self.memory = tf.TensorArray(tf.float32, 0, dynamic_size=True, clear_after_read=False,
                element_shape=[batchSize, self.numStateNodes+self.numKeyNodes], name='memTA')
        if initialMemValues is not None:
            self.memory = self.memory.unstack(initialMemValues)
        # Static conditioning array : [seq, batchSize, numStateNodes+numKeyNodes]
        # Attention is applied separately from internal memory, but with they same queries
        self.extraMemArray = extraMemArray
        self.timeOffset = self.memory.size()

    @property
    def state_size(self):
        return (self.outputSize, 1)

    @property
    def output_size(self):
        # Value, Query
        return self.outputSize

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        # The input must contain value and query. They may overlap.
        assert input_depth >= max(self.numKeyNodes, self.numStateNodes)
        kernelIn = self.numStateNodes# + self.numKeyNodes# input_depth
        kernelOut = self.numStateNodes + 1*self.numKeyNodes
        self._kernel = self.add_variable('weights', shape=[kernelIn, kernelOut])
        self._bias = self.add_variable('biases', shape=[kernelOut],
            initializer=tf.zeros_initializer(dtype=self.dtype))

        self.built = True

    def attention(self, memoryTensor, query):
        return attention(memoryTensor,
                         query,
                         self.batchSize,
                         self.numStateNodes,
                         self.numKeyNodes,
                         self.numHeads)

    def call(self, inputs, state):

        # There is one time counter per batch element. If the cell is unrolled with sequence_length
        # parameter, the time counter will stop at the provided sequence_length for each sequence
        # in the batch. We take the max here to always write to a new index of the memory array.
        # TODO: if the longest sequence length specified is shorter than the number of elements
        # in the input tensor, then this will fail since the time counter will get stuck at that
        # specified sequence_length.
        # TODO: what is wrong with self.memory.size() ?
        time = tf.to_int32(tf.reduce_max(state[1])) + self.timeOffset

        # [batchSize, numHeads*(numKeyNodes+numStateNodes)
        keyValuePair = state[0]
        self.memory = self.memory.write(time, keyValuePair)
        # [seq, batchSize, numHeads*(numKeyNodes+numStateNodes)]
        memoryTensor = self.memory.stack()
        # Keys may overlap values, so they are selected from the end using -numKeyNodes:
        attended = self.attention(memoryTensor, query=inputs[:, -self.numKeyNodes:])
        inputs = attended + inputs[:, :self.numStateNodes]
        if self.extraMemArray is not None:
#            attendedExtra = self.attention(self.extraMemArray, query=inputs[:, -self.numKeyNodes:])
            attendedExtra = self.extraMemArray
            attendedExtra = tf.reshape(attendedExtra, [-1, self.numStateNodes])
            inputs += attendedExtra
        newState = tf.matmul(inputs, self._kernel)
        newState = tf.nn.bias_add(newState, self._bias)
        if self.layerNorm:
            newState = tf.contrib.layers.layer_norm(newState, trainable=True,
                activation_fn=self.activations)
        else:
            newState = self.activations(newState)
        output = newState
        newState = (newState, state[1]+1.0)
        return output, newState

"""
batchSize = 4
stateSize = 3
keySize= 2
numHeads = 1
initialMemValues = tf.ones([3, batchSize, numHeads*(stateSize+keySize)])
cell = SelfAttentiveCell(stateSize, keySize, batchSize, numHeads, initialMemValues)
sequences = tf.ones([batchSize, 5, 6] , dtype=tf.float32)
lens = [1, 2, 3, 100]

outputs, finalState = tf.nn.dynamic_rnn(cell, sequences,  sequence_length=lens,dtype=tf.float32)

#memSize = cell.memorySize()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(outputs)
print
states = sess.run(finalState)
for s in states:
    print s
#print sess.run(memSize)
#print sess.run(cell.memorySize())
exit()
"""

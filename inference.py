import os
import sys

import numpy as np

import data_pipe

class InferenceRunner(object):
    def __init__(self, sess, model, data, context_dir, context_placeholder, max_context_len,
        groundtruth_mode, softmax_temps=[0.75], num_predictions=32, num_runs_per_call=2):
        """
        """
        self.sess = sess
        self.model = model
        self.data = data
        self.context_files =['{}/{}'.format(context_dir, f) for f in os.listdir(context_dir)]
        self.context_placeholder = context_placeholder
        self.max_context_len = max_context_len
        self.groundtruth_mode = groundtruth_mode
        self.softmax_temps = softmax_temps
        self.num_predictions = num_predictions
        self.num_runs_per_call = num_runs_per_call

    def _split_context(self, context):
        """
        Separates the context into 2 parts, so the second part can be displayed as the ground_truth.
        context: a string of text representing a conditioning context
        """
        if self.groundtruth_mode == 'half':
            # split the contexting line in half
            context = context.split()
            num_words = len(context)
            ground_truth = ' '.join(context[-num_words/2:])
            context = ' '.join(context[:num_words/2])
        elif self.groundtruth_mode == '|SEP|' and self.groundtruth_mode in context:
            # split the contexting line on instances of the delimiter: "|SEP|"
            context = context.split(self.groundtruth_mode)
            ground_truth = self.groundtruth_mode.join(context[1:])
            context = context[0]
        else:
            # use the whole context, don't split off a ground_truth
            ground_truth = 'NO GROUND TRUTH'
        return context, ground_truth

    def run_inference(self):
        for run in range(self.num_runs_per_call):
            contexts = data_pipe.getRandomSentence(self.context_files, numSamples=1)
            for context in contexts:
                context = context[0].strip()
                context, ground_truth = self._split_context(context)
                print 'Context: {}'.format(context)
                print 'Ground-truth: {}'.format(ground_truth)
                for softmax_temp in self.softmax_temps:
                    print softmax_temp
                    self._run_inference_on_context(context, softmax_temp)
                print

    def _run_inference_on_context(self, context, softmax_temp):
        """
        context: string passed asinitial condition from which model generates next tokens
        """
        result = context.strip()
        try:
            for word_idx in range(self.num_predictions):
                # TODO: Rerunning the sentence encoder over the entire history of words is wasteful,
                # should checkpoint the internal states of the SARAhs rather than trimming history
                recent_words = ' '.join(result.split()[-self.max_context_len:])
                feed = {self.context_placeholder:recent_words}
                sentences_encoded_3 = self.sess.run(self.model.sentences_encoded_checkpoint_3,
                    feed_dict=feed)
                word_vectors_2 = sentences_encoded_3[:, -1, :]
                next_word = self._run_word_decode(word_vectors_2, softmax_temp)
                print(next_word),
                sys.stdout.flush()
                result += ' ' + next_word
        except Exception as e:
            print e
        print

    def _run_word_decode(self, word_vectors_2, softmax_temp):
        word = ''
        feed = {self.model.sentences_encoded_placeholder_3:np.expand_dims(word_vectors_2, axis=1),
                self.model.softmax_temp:softmax_temp}
        for char_idx in range(self.model.config['max_word_len']):
            feed[self.context_placeholder] = word
            predictions = self.sess.run(self.model.predictions_3, feed_dict=feed)
            next_char = self.data.id_to_chr[predictions[0, 0, char_idx]]
            if next_char == self.data.go_stop_token: # End of word
                if char_idx == 0: # End of sentence
                    word += next_char
                break
            word += next_char
        return word

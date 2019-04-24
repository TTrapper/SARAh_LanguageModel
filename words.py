import argparse
from collections import Counter
import os
import pickle

import bokeh
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, TapTool, CustomJS, LabelSet
import numpy as np
import tensorflow as tf
import umap


import config
import data_pipe
import train

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--restore', type=str, required=True)
parser.add_argument('--vocab_file', type=str, default='./vocab')
parser.add_argument('--max_size', type=int, default=None)

def get_vocab(datadir, vocab_file, max_size=None, words_only=False):
    if os.path.exists(vocab_file):
        print 'loading precomputed vocab: {}'.format(vocab_file)
        word_to_freq = pickle.load(open(vocab_file, 'rb'))
    else:
        print 'computing word vocabulary from {}'.format(datadir)
        word_to_freq = Counter()
        for f in os.listdir(datadir):
            word_to_freq += Counter(open(datadir + f, 'r').read().split())
        pickle.dump(word_to_freq, open(vocab_file, 'wb'))
    word_to_freq = word_to_freq.most_common(max_size)
    return [word_freq_tuple[0] for word_freq_tuple in word_to_freq] if words_only else \
        Counter(dict(word_to_freq))

def get_word_embeds(savename, datadir, vocab, restore):
    embed_file = '{}.embeds.npy'.format(savename)
    if os.path.exists(embed_file):
        print 'loading precomputed embeddings: {}'.format(embed_file)
        word_embeds = np.load(embed_file)
        return word_embeds
    word_embeds = compute_word_embeds(datadir, restore, vocab, embed_file)
    return word_embeds

def compute_word_embeds(datadir, restore, wordlist, savepath):
    print 'computing word embeddings from model: {}'.format(restore)
    free_model, data, sess = sess_setup(datadir, restore)
    batch_size = 2048
    batched_words = [wordlist[i:i+batch_size] for i in range(0, len(wordlist), batch_size)]
    word_embeds = []
    for words in batched_words:
        words = ' '.join(words) # provide words as one long sentence (workaround to make a single batch)
        src_word_embeds_3 = sess.run(free_model.word_embeds_3, feed_dict={data.trg_place:words})
        src_word_embeds_3 = src_word_embeds_3[0, 1:, :] # Remove batch dim and GO word
        print src_word_embeds_3.shape
        word_embeds.append(src_word_embeds_3)
    word_embeds = np.concatenate(word_embeds, axis=0)
    print word_embeds.shape
    np.save(savepath, word_embeds)
    return word_embeds

def sess_setup(datadir, restore_dir):
    conf = config.generate_config(keep_prob=1.0, noise_level=0)
    data = data_pipe.Data(datadir, conf['batch_size'], conf['max_word_len'],
        conf['max_line_len'])
    _, free_model = train.build_model(data, conf)
    sess = tf.Session()
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, restore_dir)
    return free_model, data, sess

def project_embeds(savename, embeds):
    project_file = '{}.projected.npy'.format(savename)
    if os.path.exists(project_file):
        print 'loading precomputed embedding projections: {}'.format(project_file)
        projected_embeds = np.load(project_file)
    else:
        print 'projecting word embeds'
        projected_embeds = umap.UMAP().fit_transform(embeds)
        np.save(project_file, projected_embeds)
    return projected_embeds

def plot_projections(projection, labels, outfile):
    projection, labels = projection, labels
    # output to static HTML file
    output_file('{}.projected.html'.format(outfile))

    source = ColumnDataSource(data=dict(
        x=projection[:, 0],
        y=projection[:, 1],
        desc=labels))

    # Custom JS that runs when a point is clicked
    code = """
    const data = labels.data
    for (i=0; i<points.selected.indices.length; i++) {
        const ind = points.selected.indices[i]
        currentIdx = data.t.indexOf(points.data.desc[ind])
        if (currentIdx > -1) {
            data.x.splice(currentIdx, 1)
            data.y.splice(currentIdx, 1)
            data.t.splice(currentIdx, 1)
            currentIdx = data.ind.indexOf(ind)
            data.ind.splice(currentIdx, 1)
            continue;
        } else {
            data.x.push(points.data.x[ind])
            data.y.push(points.data.y[ind])
            data.t.push(points.data.desc[ind])
            data.ind.push(ind)
        }
    }
    console.log(data)
    labels.change.emit()
    """

    p = figure(title='embeds', plot_width=1000, plot_height=1000)
    p.title.text_font_size = '16pt'
    # Hover tool
    hover = HoverTool(tooltips=[("", "@desc")])
    p.add_tools(hover)
    labels = ColumnDataSource(data=dict(x=[], y=[], t=[], ind=[]))
    # Tap tool (run custom JS)
    callback=CustomJS(args=dict(points=source, labels=labels), code=code)
    tap = TapTool(callback=callback)
    p.add_layout(LabelSet(x='x', y='y', text='t', y_offset=4, x_offset=4, source=labels))
    p.add_tools(tap)
    p.circle('x', 'y', source=source, size=8)
    show(p)

if __name__ == '__main__':
    args = parser.parse_args()
    vocab = get_vocab(args.datadir, args.vocab_file, args.max_size, words_only=True)
    word_embeds = get_word_embeds(args.vocab_file, args.datadir, vocab, args.restore)
    projected_embeds = project_embeds(args.vocab_file, word_embeds)
    plot_projections(projected_embeds, vocab, args.vocab_file)



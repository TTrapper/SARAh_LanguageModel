import argparse
import os
import random
import re
import sys
import unicodedata

from nltk.tokenize.punkt import PunktSentenceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--sentences', type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--newline', type=str, default=' _-_ ')

def parse_by_alpha(line):
    """
    Parses by whitespace, then each word is split into groups of alphas and non-alphas
    ex: "don't do that please.": [don ' t do that please .]
        "http://www.bs.com/184": [http :// www . bs . com /184]
    """
    words = []
    for word in line.strip().split():
        words.extend(re.split('([^a-zA-Z]+)', word))
    # The above splits add null strings, remove them.
    return [word for word in words if word != '']

def _process_dir(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_docs = os.listdir(src_dir)
    for doc_num, src_doc in enumerate(src_docs):
        print 'processing documents ({} of {})\r'.format(1 + doc_num, len(src_docs)),
        sys.stdout.flush()
        src_filepath = '{}/{}'.format(src_dir, src_doc)
        dst_filepath = '{}/{}'.format(dst_dir, src_doc)
        _process_doc(src_filepath, dst_filepath)

def _process_doc(src_filepath, dst_filepath):
    with open(dst_filepath, 'w') as dst_doc, open(src_filepath, 'r') as src_doc:
        src_doc = src_doc.read()
        src_doc = src_doc.replace('\n', args.newline).replace('\r', args.newline)
        if args.sentences: # Parse doc into sentences
            src_doc = src_doc.decode('utf-8', 'ignore')
            tokenizer = PunktSentenceTokenizer(src_doc)
            src_doc = tokenizer.tokenize(src_doc)
            src_doc = [' '.join(parse_by_alpha(s)) for s in src_doc]
            src_doc = [s.encode('utf-8', 'ignore') for s in src_doc]
        else: # Parse doc into fixed number of words per line
            src_doc = parse_by_alpha(src_doc)
            src_doc = [' '.join(src_doc[i:i + args.seq_len]) for i in xrange(0, len(src_doc), args.seq_len)]
        dst_doc.write('\n'.join(src_doc))

if __name__ == '__main__':
    args = parser.parse_args()
    args.sentences = args.sentences == 'yes'
    assert(args.src != args.dst)
    _process_dir(args.src, args.dst)

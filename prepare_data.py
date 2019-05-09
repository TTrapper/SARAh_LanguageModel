import argparse
import os
import random
import re
import unicodedata

from nltk.tokenize.punkt import PunktSentenceTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--seq_len', type=int, default=64)
parser.add_argument('--parse_sentences', type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--autoencode', type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--go_stop_chr', type=str, default=chr(0))
parser.add_argument('--newline', type=str, default=' _-_ ')
parser.add_argument('--shuffle', type=str, choices=['yes', 'no'], default='no')

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

if __name__ == '__main__':
    args = parser.parse_args()
    args.parse_sentences = args.parse_sentences == 'yes'
    args.autoencode = args.autoencode == 'yes'
    args.shuffle = args.shuffle == 'yes'
    assert(args.src != args.dst)
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    for src_doc in os.listdir(args.src):
        with open('{}/{}'.format(args.dst, src_doc), 'w') as dst_doc, open('{}/{}'.format(args.src, src_doc), 'r') as src_doc:
            src_doc = src_doc.read()
            if args.go_stop_chr in src_doc:
                raise ValueError("The token reserved for EOS/SOS was found in the doc")
            src_doc = src_doc.replace('\n', args.newline).replace('\r', args.newline)
            if args.parse_sentences:
                src_doc = src_doc.decode('utf-8', 'ignore')
                tokenizer = PunktSentenceTokenizer(src_doc)
                src_doc = tokenizer.tokenize(src_doc)
                src_doc = [' '.join(parse_by_alpha(s)) for s in src_doc]
                src_doc = [s.encode('utf-8', 'ignore') for s in src_doc]
            else:
                src_doc = parse_by_alpha(src_doc)
                src_doc = [' '.join(src_doc[i:i + args.seq_len]) for i in xrange(0, len(src_doc), args.seq_len)]
            if args.autoencode:
                src_doc = ['{} |SEP| {}'.format(s,s) for s in src_doc] # autoencode mode repeats each sentence/line with a separator
            if args.shuffle:
                random.shuffle(src_doc)
            dst_doc.write('\n'.join(src_doc))

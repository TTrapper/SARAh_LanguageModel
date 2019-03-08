import argparse
import os
import re
import unicodedata

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True)
parser.add_argument('--dst', type=str, required=True)
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--newline_replacement', type=str, default=' ][ ')
args = parser.parse_args()
assert(args.src != args.dst)

if not os.path.exists(args.dst):
    os.makedirs(args.dst)

for src_doc in os.listdir(args.src):
    with open('{}/{}'.format(args.dst, src_doc), 'wb') as dst_doc, open('{}/{}'.format(args.src, src_doc), 'r') as src_doc:
        src_doc = src_doc.read().decode('utf-8').encode('ascii', 'replace') # TODO Loss of information on ascii encode
        src_doc = src_doc.replace('\n', args.newline_replacement).replace('\r', args.newline_replacement)
        # Split doc into seq_len sized chunks
        num_chars_range = '.{{{},{}}}'.format(1, args.seq_len)
        src_doc = re.findall(num_chars_range, src_doc)
        # Pad the last string to seq_len
        src_doc[-1] = src_doc[-1].ljust(args.seq_len)
        dst_doc.write('\n'.join(src_doc))

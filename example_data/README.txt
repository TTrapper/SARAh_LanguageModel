Data files are assumed to be byte-encoded raw text.
Data files are assumed to be in a directory containing only other data files (such as this example_data folder). 

Data should be preprocessed by the prepare_data script.
For example, to parse words and generate files with a fixed number of words per line, use (from the root directory):
python prepare_data.py --src ./example_data --dst ./preprocessed_example_data
To extract sentences rather than fixed-length lines, use --parse_sentences yes

Note that during model training, the data pipeline further processes the text:
Lines may be clipped to a maximum length (number of words). This length is specified in the configuration file.
Words may be clipped to a maximum length (number of characters). This length is specified in the configuration file.

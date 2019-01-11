Data should be ascii or otherwise encode 1 byte per character: Tensorflow's string_split doesn't play nice otherwise.
Words are assumed to be be delimited by a single space character.
The source/target pairs are selected by choosing concurrent lines, so each line is considered the target for the previous line.
Lines may be clipped to a maximum length (number of words). This length is specified in the configuration file.
Words may be clipped to a maximum length (number of characters). This length is specified in the configuration file.

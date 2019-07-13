# Paraphrases from a Self-Supervised Language Model
## Overview
This is an implementation of an attention-based language model, along with some tools to get it to generate paraphrases (or at least semantically similar sentences), in a fully self-supervised way.

This is similar to work I completed for my Masters Thesis:

https://dalspace.library.dal.ca/handle/10222/75034

Once trained, the model generates a list of paraphrases for a given input sentence.  For example:

**Input:**
_The rain fell from the sky._

**Outputs:**
- _Rain poured from the sky._
- _Rain poured out of the sky._
- _Thunder rolled and lightning flew._

**Input:**
_The thunder roared._

**Outputs:**
- _The thunder crackled again._
- _The thunder boomed._
- _The thunder rumbled._
- _The rumble of thunder from the shooting plane echoed along the rooftops._


**Input:**
_The sun shone orange in the blue sky._

**Outputs:**
- _The sun shone overhead._
- _The sun shone bright orange on the horizon._
- _The sun shone in the black sky._

**Input:**
_It was almost dawn when he pulled into our driveway, the sky just beginning to tinge pink at the edges._

**Outputs:**
- _I was still on the phone when he came to a stop, a cloud of dust rising from his footstool._
- _I felt the heat of his gaze against my own, the warmth of the sun shining down on us._
- _I walked to the bus stop , and I stared out the window , watching the sun slowly climb over the horizon._



### How it works
The basic strategy is as follows:
1) Pretrain the language model on a large corpus of text
2) Extract sentence representations from the model (fixed-size representations are learned implicitly)
3) Use these sentence representations to find nearby, semantically similar sentences
4) Create a text dataset where each line is the concatenation of sentences whose encodings are nearby
5) Fine-tune the model on this new dataset
6) Enjoy your self-surpervised paraphrase generator

## Model
The model architecture is hierarchical and consists of 3 parts:
1. word-encoder
2. context-encoder
3. word-decoder

### Word Encoder
The word encoder looks up an embedding vector for each character in the word. The char embeddings are then modified with their relative position within the word and summed together to create a word vector. 

### Context Encoder
The context encoder is an attention-based recurrent model, which I have called the Self Attentive Recurrent Array. It is related to Transformers and Neural Turing Machines. It is just a feed forward layer with an array of previous values. When presented with a word vector, it uses multi-head attention to combine the current word with previous output values, then computes a new output vector and appends it to the memory array.

### Word Decoder
The word decoder predicts the next word in the stream using the output of the context encoder as a conditioning vector. It predicts each character of the word from the concatenated embeddings of previous characters. 

## Usage
todo

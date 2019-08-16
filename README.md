# Paraphrases from a Self-Supervised Language Model
## Overview
This is an implementation of an attention-based language model, along with some tools to get it to generate paraphrases (or at least semantically similar sentences), in a fully self-supervised way.

This is similar to work I completed for my Masters Thesis:

https://dalspace.library.dal.ca/handle/10222/75034

Once trained, the model generates a list of "paraphrases" for a given input sentence. The paraphrase qualiy is better for concepts that are common in the training set, such as short descriptions of weather. For more complex inputs it may not paraphrase exactly, but it still manages to replicate the semantic flavour of the inputs. Sometimes it gets lucky and produces somethings rather insightful. For example:

**Input:**
_Earth is the origin of the human species._

**Output:** _It is the original computer system of the gene pool._


## Exampes:

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
_She walked through the forest with her dog._

**Outputs:**
- _She walked in the woods with her dog._
- _She walked along behind the little animal._ 
- _She ran through the broken trees like an owl._

**Input:**
_A being of pure light that moved like water._                                               

**Outputs:**
- _A soul that was a star in the sky._
- _A sickening spirit that was beyond human._
- _A shining fire that floated in the air._ 

**Input:**
_Four candles lit the room in the far corner._

**Outputs:**
- _Two more candles illuminated the room._
- _Two red beams lit the room._
- _Two glass lamps hung from the window._
- _Three ornate candles flickered in the hall._

**Input:**
_The ocean glittered with moonlight._

**Outputs:**
- _The sea shimmered with light._
- _The sea seemed to glow in the sun._
- _The sea was still and shining._

**Input:**
_The book was old and smelled of must and mildew._

**Outputs:**
- _It was strange and stained with dust and decay._
- _The pantry was clean and spiced with the first leaves._
- _It was young and fed well._

**Input:**
_Her smile left me speechless._

**Outputs:**
- _Her smile left me stunned._
- _Her smile left me shocked._
- _Her smile only made my heart beat a little faster._

**Input:**
_The crow stood black against a blue sky._

**Outputs:**
- _The beast was blue and pale._
- _The creature was black as a star._
- _The winged creature was black as a star ._

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

# SARAh_LanguageModel
This is an implementation of an attention-based language model, along with some tools to get it to generate paraphrases (or at least semantically similar sentences), in a fully self-supervised way.

This is similar to work I completed for my Masters Thesis:

https://dalspace.library.dal.ca/handle/10222/75034

Once trained, the model can generate paraphrases (or at least semantically similar sentences).  For example:

**Input:**
The rain fell from the sky.

**Outputs:**
- Rain poured from the sky.
- Rain poured out of the sky .
- Thunder rolled and lightning flew.

**Input:**
The thunder roared.

**Outputs:**
- The thunder crackled again.
- The thunder boomed.
- The thunder rumbled.
- The rumble of thunder from the shooting plane echoed along the rooftops.


**Input:**
The sun shone orange in the blue sky.

**Outputs:**
- The sun shone overhead. 
- The sun shone bright orange on the horizon. 
- The sun shone in the black sky.

**Input:**
It was almost dawn when he pulled into our driveway, the sky just beginning to tinge pink at the edges.

**Outputs:**
- I was still on the phone when he came to a stop, a cloud of dust rising from his footstool. 
- I felt the heat of his gaze against my own, the warmth of the sun shining down on us. 
- I walked to the bus stop , and I stared out the window , watching the sun slowly climb over the horizon.



**How it works**
The basic strategy is as follows:
1) Pretrain the language model on a large corpus of text
2) Extract sentence representations from the model
   - The model learns to encode semantically similar sentences near each other
   - For this I simply use the model's output at the last word of the sentence
3) Use the extracted sentence representations to find semantically similar sentences
4) Create a dataset where each line (training example) is a tuple of similar sentences
5) Fine-tune the model on this new dataset
6) Enjoy your self-surpervised paraphrase generator.

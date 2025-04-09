# Transformers!

## Why transformers? 
The shift from RNNs/LSTMs came because of two major issues: vanishing/exploding gradients and a lack of a large context window. These disadvantages are due to the fact that these architectures move through a sentence SEQUENTIALLY - So during backpropagation, the gradient is calculated many, many times. If the gradient value (dL/dTheta) is less than 1, then multiplying them together over and over again will cause the gradient to approach 0. In the other case, if the value is above 1, then multiplying it a bunch of times will cause it to "explode" or become very large. Transformers solve these issues!


First, we start with converting the text into vectors. Transformers account for both CONTENT and POSITIONAL embeddings for each token, so the embeddings hold both the token's information and its position within the sentence. Let's get to the interesting stuff. 

Moving quickly forward, let's get to the core of the transformer architecture: **ATTENTION**. This is the core of the GenAI boom, and what drives all the models we use like BERT and GPT. We'll start with single-head attention and then move to multihead.

## Single-Head Attention

First, we must create **Key, Query, and Value vectors** out of the input embeddings.

At a high level, here's a description of each:

**Query:** The selected token which we compare to every other token.
**Key:** A measure of how relevant every other token is to our query. 
**Value:** The actual information embedded in each token.

These are calculated through the equations equations below, where each W is a parameter matrix. It is initialized randomly, and learns over time like a traditional NN. 

<img width="374" alt="Screenshot 2025-04-09 at 12 15 53 PM" src="https://github.com/user-attachments/assets/8ca2be76-ca8b-444f-a4d2-abd79c3af5cc" />

**Now, we take the dot product of the query vector with every key vector**. This gives us an attention score, or how token _i_ should play attention to token _j_. Since every query is multiplied with every key, this is why transformers, given unlimited compute, have an infinite reference window!

In order to ensure we don't have the exploding/vanishing gradient issue, we'll do a little bit of normalization. We divide the query/key dot product by the square root of the dimensions, and take a softmax of this result to ensure the attention scores are normalized between 0 - 1. 

So far, we've reviewed everything in the single-head attention equation except for "V":

![image](https://github.com/user-attachments/assets/6b2ea70d-1d0b-40b0-b49a-3955229fe8a2)  

Now that we have a probability distribution (thanks to softmax) over every query-key pair, we must involve the actual content of each word by using the value vector we made earlier. Each token uses its attention distribution to perform a **weighted sum** over all the value vectors in the sequence. In other words, every token builds a new, context-aware representation by gathering information from all other tokens, weighted by how much attention it pays to them. This weighted sum becomes the token’s new representation - one that now encodes information from all the other tokens it attended to. Again, this is matrix multiplication 😄.

We're done! Let's apply this to multi-head attention!

## Multi-Head Attention

We'll now apply our knowledge of single-head attention to what Transformers actually use in practice.

Instead of doing just one single-head attention, we do multiple heads in parallel, each with its own learned Q, K, V projection matrices. Each head is of smaller dimension, and so it's able to focus on different aspects of the relationships between words because each head can focus on a different subspace of the input representation. A simplified example of this is that we have 4 heads where each one is reduced to work in the syntax, tone, diction, and punctuation dimensions, respectfully. This would allow each head to focus on that specific aspect alone.

After each head independently produces its own output, the outputs are concatenated and passed through a final linear layer to produce the full attention output. This allows the model to combine multiple perspectives of the text to form rich, context-aware representation for each token.

So, in essence, **multi-head attention = several single-head attentions**, each with its own perspective.


## Overall Architecture:

Attention is the most dense and important feature of transformers, but for the sake of completion, I will review the entire pipeline from _Attention is All You Need_.

![image](https://github.com/user-attachments/assets/1c635b42-70bc-44dd-8598-9544ade81ffe)

Every "Add & Norm" block that you see, the original input is added back in, and it is again normalized. This has two benefits:

1. It helps avoid the RNN exploding/vanishing gradient issue, especially when stacking many layers. <-- This is what "Nx" is. Nx stacks the "block" a constant (usually 6) number of times.
2. It lets the model preserve useful information from earlier layers.

FFNs are feed-forward neural networks which are used to add non-linearity beyond just attention. Think of it like giving each token its own little mini neural network to process its meaning after gathering context from attention.

One last thing. Let's cover the two blocks in the diagram: **Encoders and Decoders:**

- Encoder - the block which reads, parses, and encodes the input. 
- Decoder - the block which generates the output, one token at a time.

Imagine the encoder as someone reading and understanding a full question. The decoder is the person generating an answer — but can only see what they’ve already said, and they occasionally glance at the question (encoder output) for context as they form the next word.

**Attention is All You Need😎**






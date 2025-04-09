### Reinforcement Learning with Human Feedback: Process and Math.

## First, I'll describe RLHF at a high level.

Our goal is to improve the performance of a pre-trained LLM by incorporating human-feedback into the training process. This will allow us to produce outputs that align with human preferences and values.

We begin with a pre-trained LLM (e.g. GPT-2)

# Step 1: Create Preference Dataset
We ask the LLM (called the policy) to produce responses based on what we want to fine-tune it for. For example, if we wanted to create an LLM to generate summary articles that human's find interesting, we would ask the LLM to produce multiple summaries for each article. Then, we'd get a human to choose which summary they prefer for each article. We've now created a preference dataset.

# Step 2: Reward Model Training
We train another **separate** transformer model on the preference dataset. This gives us a model that can predict which responses human would prefer (or summaries humans are more likely to enjoy in our example).

# Step 3: Fine-Tuning
Using the reward model, we train the policy model (the original LLM). The policy produces new texts/summaries and our reward model "scores" them. Then, our policy is optimized to maximize its score, and the process is repeated. That's everything in a RLHF pipeline!

### Now, the math behind it all: 

## Step by step:

# Training the reward model:

We must establish a pair-wise ranking loss function; given responses y1 and y2 to a prompt x where y1 is preferred over y2 (by a human labeler), we maximize our probability that preferred response y1 is given a larger reward than the poor response y2. This is the pair-wise ranking loss function mathematically: 
<img width="353" alt="Screenshot 2025-03-13 at 6 54 58 PM" src="https://github.com/user-attachments/assets/95ce23c5-9fbd-42b6-b41d-29df5c1a9fd5" />

where r is the current reward model, theta is the current parameters, and sigmoid is a softmax function to position the value between 0 and 1. The negative log function is commonly used in binary classification problems because it penalizes incorrect predictions more heavily as the confidence in those predictions increases.

Once we calculate the loss, we'll update the parameters in order to minimize the loss. We will acheive this through gradient descent. Here is the equation:
<img width="411" alt="Screenshot 2025-03-13 at 10 39 26 PM" src="https://github.com/user-attachments/assets/a4475322-77a8-4b19-a698-1b7ca90b4eae" />

## Transformers!

# Why transformers? 
The shift from RNNs/LSTMs came because of two major issues: vanishing/exploding gradients and a lack of a large context window. These disadvantages are due to the fact that these architectures move through a sentence SEQUENTIALLY - So during backpropagation, the gradient is calculated many, many times. If the gradient value (dL/dTheta) is less than 1, then multiplying them together over and over again will cause the gradient to approach 0. In the other case, if the value is above 1, then multiplying it a bunch of times will cause it to "explode" or become very large. Transformers solve these issues!


First, we start with converting the text into vectors. Transformers account for both CONTENT and POSITIONAL embeddings for each token, so the embeddings hold both the token's information and its position within the sentence. Let's get to the interesting stuff. 

Moving quickly forward, let's get to the core of the transformer architecture: **ATTENTION**. This is the core of the GenAI boom, and what drives all the models we use like BERT and GPT. We'll start with single-head attention and then move to multihead.

# Single-Head Attention

First, we must create **Key, Query, and Value vectors** out of the input embeddings.

At a high level, here's a descrption of each:

**Query:** The token that we are comparing to every other token.
**Key:** A measure of how relevant every other token is to our key. 
**Value:** The actual information embedded in each token.

We calculate these through these equations, where each W is a parameter matrix. It is initalized randomly, and learns over time like a traditional NN. 

<img width="374" alt="Screenshot 2025-04-09 at 12 15 53 PM" src="https://github.com/user-attachments/assets/8ca2be76-ca8b-444f-a4d2-abd79c3af5cc" />




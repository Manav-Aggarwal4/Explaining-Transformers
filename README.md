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

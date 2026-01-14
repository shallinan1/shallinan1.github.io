# Transformers

This is a review file for natural language processing and transformers.

# Background

## Math Foundations

### Linear Algebra

1) What is the dot product and what does it represent geometrically? How does this relate to "similarity"?
2) If we have a matrix of dimensions N×D and multiply it by a matrix D×M, what are the dimensions of the result? Why does this matter for neural network layer design?
3) Why is matrix multiplication the core operation in neural networks? What is its computational complexity?
4) What are eigenvalues/eigenvectors and why do they matter for understanding weight matrices?
5) What is the rank of a matrix? How does low-rank approximation relate to model compression?

### Calculus and Optimization

1) What is a derivative/partial derivative and what does it represent geometrically?
2) What is the gradient and how does it relate to the direction of steepest ascent?
3) Derive the chain rule and show why it enables backpropagation.
4) Why can gradient descent get stuck in local minima, and what techniques help escape them?

### Probability and Statistics

1) What is a probability distribution? What properties must it satisfy?
2) Derive Bayes' theorem from the definition of conditional probability.
3) What is the maximum likelihood estimation (MLE) framework? Derive the MLE for a Gaussian distribution.
4) How does the softmax function convert a list of raw logits into probabilities that sum to 1? Derive it from the Boltzmann distribution or maximum entropy principle.
5) What is entropy? Derive the cross-entropy loss from the KL divergence between two distributions.
6) What is the relationship between minimizing cross-entropy and maximizing likelihood?

## ML Fundamentals

### Linear Regression

1) Derive the closed-form solution for linear regression using the normal equations.
2) Why is MSE the "right" loss for regression? (Hint: connect to Gaussian MLE)
3) What is the bias-variance tradeoff? How does model complexity affect each?

### Logistic Regression

1) Why can't we use linear regression for classification? Derive the logistic function as the solution.
2) Derive the cross-entropy loss from the MLE of a Bernoulli distribution.
3) Why is there no closed-form solution for logistic regression?

### Neural Networks

1) What is a fully connected (dense) layer? How does it transform its input?
2) Why are nonlinear activation functions (ReLU, GELU) necessary? What happens with only linear layers?
3) Derive backpropagation for a simple 2-layer network. What is the computational complexity?
4) What causes the vanishing/exploding gradient problem? How do ReLU and residual connections help?
5) What are residual (skip) connections and how do they help information and gradients flow through deep networks?
6) What is the universal approximation theorem and what are its practical limitations?

#### Graph Neural Networks

1) Derive the message passing update rule. How do nodes aggregate information from neighbors?
2) What is over-smoothing in deep GNNs? Why does it happen?
3) How do you handle variable-sized graphs in batched training?

### Optimization Algorithms

1) Why does SGD work despite using noisy gradient estimates? What role does the noise play?
2) Derive the momentum update rule. Why does it help with saddle points and narrow valleys?
3) Explain Adam: how does it combine momentum with adaptive learning rates? When does AdamW help?
4) What is the learning rate warmup and why is it important for transformers?

### Regularization

1) Derive why L2 regularization is equivalent to a Gaussian prior on weights (MAP estimation).
2) Why does dropout work? Explain the "ensemble" interpretation vs the "noise injection" interpretation.
3) Derive the batch normalization forward pass. Why does it help with internal covariate shift?
4) Why do transformers use layer norm instead of batch norm? Where is it placed (pre-LN vs post-LN)?

## NLP Foundations

### Tokenization

1) What are the tradeoffs between word-level, character-level, and subword tokenization?
2) Explain the BPE algorithm. How does it balance vocabulary size vs sequence length?
3) How do tokenization choices affect model performance on rare words, morphology, and multilinguality?

### Word Embeddings

1) What is a word embedding? Why is a dense vector better than a one-hot encoding?
2) Derive the skip-gram objective. What does the dot product between embeddings represent?
3) Show that GloVe is implicitly factorizing a co-occurrence matrix. How does this relate to skip-gram?
4) Why do static embeddings fail on polysemy? Give examples where this matters.

### Language Modeling

1) What is the autoregressive factorization of P(x)? How does this connect to cross-entropy loss?
2) What is perplexity? Derive it from cross-entropy. What does a perplexity of 100 mean intuitively?
3) What is the exposure bias problem with teacher forcing? How do scheduled sampling and other techniques address it?

## Sequential Models

### RNNs and LSTMs

1) How do RNNs process sequential data? Why does sequential processing create a computational bottleneck?
2) What is the long-term dependency problem? Why do RNNs "forget" the beginning of long sequences?
3) Derive why vanilla RNNs have vanishing gradients by unrolling through time.
4) Explain each LSTM gate. Why does the additive cell state update help gradient flow?

### Sequence-to-Sequence Models

1) What is the information bottleneck problem with fixed-size context vectors?
2) Derive beam search. Why is it better than greedy decoding but still suboptimal?
3) What is the length bias problem in seq2seq? How is it typically addressed?

### Attention (Pre-Transformer)

1) If we can't process words left-to-right, how else could we determine which words are related to each other?
2) What does it mean to "attend" to specific parts of an input sequence while generating an output?
3) Derive Bahdanau (additive) attention. What are the query, key, and value in this formulation?
4) What is Luong (multiplicative) attention and how does it differ?
5) How does attention solve the information bottleneck? What new bottleneck does it create (hint: complexity)?
6) If we process all words in parallel (not sequentially), how can we encode their position or order?



# Basic Transformer Questions (High Level)

1) What is the high-level idea of how transformers work?

   <details><summary>Answer</summary>
    Test
   </details>

TODO: more questions here, loss funciton

# Contemporary Questions
TODO: add questions like "what does flash attention do differrently"


# Historical Questions

TODO: add questions like "why do we do this instead of this architechture, etc"
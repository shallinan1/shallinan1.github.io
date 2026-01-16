# Transformers

This is a review file for natural language processing and transformers.

# Background

## Math Foundations

### Linear Algebra

#### Vector Knowledge
1) What is the dot product and what does it represent geometrically?

    The dot product represents the scaled overlap of two vectors. It is a scalar, unbounded, real value. Geometrically, it calculates how much of Vector A points in the direction of Vector B, and then multiplies that amount by the total length of Vector B. 
 
    It can be computed as the sum of the element-wise product of two vectors, or as $a \cdot b = ||a|| \ ||b|| \ cos (\theta)$. Intuitively, this tells us not just if they point in the same direction, but how much "energy" or magnitude they share in that common direction.
    
    The dot product is 0 when the vectors are *orthogonal*. A positive dot-product means that they point in the same direction, while negative value means they point in opposite directions. Importantly, the dot product conflates direction AND magnitude, which is why you can't tell from a single value whether it's large because of alignment or because the vectors are long.
    
    See this good resource for more diving in: [Better Explained - Understanding the Dot Product](https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/)

2) What is cosine similarity? Derive it from the dot product formula and explain how to interpret its values.

    Cosine similarity refers to taking cosine of the angle between two vectors; it is a bounded scalar $\in [-1, 1]$. Intuitively, it tells you how similar the *direction* of two vectors are in n-dimensional space. We can rearrange the dot product equation to get: 
    $$\cos (\theta) = \frac{a \cdot b}{||a|| \ ||b||}$$
    So the cosine similarity is just the dot product of two vectors normalized by their magnitude! Like above, a value of 0 indicates orthogonality, while positive and negative signs represent pointing in the same and opposite direction respectively.

    **Note**: Since we normalize, the magnitude of the vectors being compared does not matter, only the direction!

3) What is a vector projection? Show how the dot product a · b can be interpreted as the projection of a onto b, scaled by ||b||.

    A vector projection of $a$ onto $b$ is an *orthogonal projection* of $a$ onto a straight line parallel to $b$. Intuitively, it is the resultant vector if we draw a perpendicular line from the end of $a$ to the line $b$ sits on. See the figure for a visual. Another analogy is that the projection is the "shadow" $a$ casts onto $b$.

    Let $\hat{b} = \frac{b}{||b||}$ be the unit vector in the direction of b. The projection points in this direction. Its length comes from basic trigonometry: $||a|| \cos(\theta)$ gives the adjacent side (the component of a along b). Multiply direction by length to get the vector projection: $$\text{proj}_b(a) = ||a|| \cos(\theta) \cdot \hat{b}$$

    **Note on sign**: When θ > 90°, cos(θ) < 0, so the projection points opposite to b. This matches the dot product's sign behavior. 

    The **scalar projection** of a onto b is just the signed length of the above projection.
    $$\text{comp}_b(a) = ||a|| \cos(\theta)$$

    This formula should look familiar. Recall that $ a \cdot b = ||a|| \ ||b|| \cos(\theta)$. Then, we see that $a \cdot b = \text{comp}_b(a) \times ||b||$. Intuitively, the dot product is the scalar projection of a onto b scaled by the length of b. Interestingly, we also see that the flipped case is also true $a \cdot b = \text{comp}_a(b) \times ||a||$. 

    <p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Projection_and_rejection.svg/1280px-Projection_and_rejection.svg.png" alt="Projection of a on b (a1), and rejection of a from b (a2)" width="25%">
    <br>
    <em>Projection of a on b (a1), and rejection of a from b (a2)</em>
    </p>

4) What is a vector norm? What do L1 (Manhattan) and L2 (Euclidean) norms measure geometrically?

    The $L_p$ norm of a vector $v$ is a non-negative real value:

    $$||v||_p = \left(\sum_{i=1}^{n} |v_i|^p \right)^{1/p}$$

    Geometrically, the L1 norm represents the "taxicab" distance from the origin to the vector, while the L2 norm represents the magnitude or length of the vector, ie, the "straight-line" or Euclidean distance.

    TODO why is the abs before the power

5) What happens as p increases (L3, L4, ... L∞)? What does the L∞ norm measure?

    As p increases, the norm gives more weight to larger components. In the limit:

    $$||v||_\infty = \max_i |v_i|$$

    The $L_∞$ norm is just the largest absolute value in the vector. Intuitively, higher p asks "how big is the biggest element?" rather than considering all elements equally.

    **Unit ball visualization** (all vectors with norm ≤ 1 in 2D):
    - $L_1$: Diamond
    - $L_2$: Circle
    - Higher $p$: Corners get rounder, approaching a square
    - $L_\infty$: Square

6) What are the defining properties of a norm?

    1. Non-negativity: $||v|| \geq 0$, equals 0 iff $v = 0$
    2. Absolute homogeneity: $||cv|| = |c| \times ||v||$
    3. Triangle inequality: $||u + v|| \leq ||u|| + ||v||$ (TODO can explore these with more questions)
    
7) How does dividing by the $L_2$ norm create a unit vector? Why is this called "normalization"?

    Let $\hat{v} = \frac{v}{||v||_2}$. Using absolute homogeneity:

    $$||\hat{v}||_2 = \left|\left|\frac{v}{||v||_2}\right|\right|_2 = \frac{1}{||v||_2} \times ||v||_2 = 1$$

    The result has $L_2$ norm (length) equal to 1. This is called "normalization" because we're scaling the vector to have a standard (normal) length, preserving only direction.

    **Note**: Dividing by any $L_p$ norm creates a unit vector in that norm's sense. "Unit vector" conventionally means $L_2$ norm = 1.

#### Matrices

1) If we have a matrix of dimensions N×D and multiply it by a matrix D×M, what are the dimensions of the result? Why does this matter for neural network layer design?

    The dimensions will be N×M. Specifically, there will be N rows (from A) × M columns (from B) = N×M dot products = N×M elements. This matters in neural network design because we need to ensure the output dimension of each layer aligns with the input dimension of the next layer.

2) Why is matrix multiplication the core operation in neural networks? What is its computational complexity?

    Matrix multiplication is the core operation because each layer computes a linear transformation of its inputs—and linear transformations *are* matrix multiplications.

    Multiplying an $(N \times D)$ matrix by a $(D \times M)$ matrix produces an $(N \times M)$ result—that's $N \times M$ dot products of length-$D$ vectors. This requires $N \times D \times M$ multiply-accumulate operations, giving $O(NDM)$ time complexity.

    In neural networks, one matrix is a batch of inputs ($N$ samples, $D$ features each), and the other is weights ($D \times M$, where $M$ is the number of neurons). Each column of the weight matrix encodes one neuron's learned pattern; each dot product gives one neuron's *activation* for one sample. Remember when we thought about intuitive explanations for the dot product? This is one of them: a high activation means the input shares a lot of energy in the direction the neuron's weights are pointing, so the neuron "fires" strongly. One matrix multiply computes all activations for all samples simultaneously.

    TODO talk a bit how these activations are fed into the next layer
    TODO summation format of the matrix multiplication

3) What is a transpose? When do you need to transpose a matrix before multiplying?

    A transpose flips a matrix across its main diagonal: rows become columns and columns become rows. An $(N \times D)$ matrix becomes $(D \times N)$.

    For matrix multiplication, the inner dimensions must match: $(N \times \mathbf{D}) \cdot (\mathbf{D} \times M)$ works, but $(N \times D) \cdot (N \times D)$ doesn't. You transpose when you need to reuse the same matrix in a different orientation.

4) What is the rank of a matrix? How does low-rank approximation relate to model compression (e.g., LoRA)?

    The rank of a matrix is the number of linearly independent rows (or columns; they're equal). An $(N \times D)$ matrix has rank at most $\min(N, D)$. Why does this matter? Linearly dependent rows can be reconstructed from other rows, so you don't need to store them explicitly.

    Example: If a $(1000 \times 1000)$ matrix has rank 50, it looks big but only does "50 dimensions of work." The other 950 rows are combinations of those 50. This doesn't change how the network computes (each neuron still fires independently), but it means the weight matrix can be stored more efficiently. You can factor this into $(1000 \times 50) \cdot (50 \times 1000)$: the second matrix holds the 50 basis rows, and the first matrix holds coefficients for how to combine them to reconstruct each of the 1000 original rows. Storage drops from 1,000,000 to 100,000 values.

    LoRA (Low-Rank Adaptation) exploits this for fine-tuning. Instead of updating a full weight matrix $W$ with shape $(D \times M)$, you learn a low-rank update:

    $$W' = W + BA$$

    where $B$ is $(D \times r)$ and $A$ is $(r \times M)$, with $r \ll D, M$. The low-rank constraint is structural: any $(D \times r) \cdot (r \times M)$ product has rank at most $r$ by construction. No regularization needed. Why does this work? Empirically, weight updates during fine-tuning tend to lie in a low-dimensional subspace. You get most of the adaptation benefit with a fraction of the trainable parameters.

5) What are eigenvalues/eigenvectors? How do they relate to rank, and why does the condition number (ratio of largest to smallest eigenvalue) matter for training stability?

    For a square matrix $A$, an eigenvector $v$ is a direction that only gets scaled (not rotated) when $A$ is applied: $Av = \lambda v$. The eigenvalue $\lambda$ is the scaling factor. Most vectors get both stretched and rotated; eigenvectors are the special directions where the matrix just stretches or compresses.

    **Relation to rank:** The rank equals the number of non-zero eigenvalues (counting with multiplicity, so repeated eigenvalues count multiple times). An eigenvalue of 0 means that direction gets collapsed to nothing.

    **Condition number** is the ratio of largest to smallest eigenvalue: $\kappa = |\lambda_{max}| / |\lambda_{min}|$.

    Why it matters for training: the Hessian matrix (second derivatives of the loss) tells you the curvature of the loss landscape. Its condition number measures how different the steepest vs shallowest directions are. High condition number means a long narrow valley: one learning rate can't handle both directions well.

    **Note:** Eigenvalues are only defined for square matrices. For non-square matrices (most weight matrices), singular values from SVD serve the same role. The condition number becomes $\sigma_{max} / \sigma_{min}$, and the training stability intuition is identical.

    TODO other deeper knowledge about the Hessian

### Calculus and Optimization

1) What is a derivative/partial derivative and what does it represent geometrically?

    A derivative tells us the slope of a function with respect to a specific variable. Geometrically, it tells us an equation for the slope of the line tangent to the function with respect to a variable.

2) What is the gradient and how does it relate to the direction of steepest ascent?

    The gradient is the derivative (?). It points in the direction.

3) Derive the chain rule and show why it enables backpropagation.

    The chain rule tells us that $\frac{\partial x}{\partial z} =\frac{\partial x}{\partial y} \times \frac{\partial y}{\partial z}$.

4) Why can gradient descent get stuck in local minima, and what techniques help escape them?

    When the partial derivative is 0, then we get stuck.

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
4) In the backward pass, why do we multiply by $W^T$ instead of $W$? (Hint: think about "fanning out" vs "fanning in")

    Concrete example in backpropagation:
    - **Forward pass**: $Y = XW$ where $X$ is $(N \times D)$ and $W$ is $(D \times M)$
    - **Backward pass**: you receive gradient $\frac{\partial L}{\partial Y}$ with shape $(N \times M)$ and need gradient $\frac{\partial L}{\partial X}$ with shape $(N \times D)$
    - To get there: $(N \times M) \cdot (M \times D) = (N \times D)$, so you need $W^T$

    The weights "fan out" from $D$ inputs to $M$ neurons in the forward pass. To propagate gradients backward, you "fan in" from $M$ neurons back to $D$ inputs. Same weights, opposite direction.
    
5) What causes the vanishing/exploding gradient problem? How do ReLU and residual connections help?
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
2) What's the difference between L1 and L2 regularization? Why does L1 promote sparsity (drive weights to exactly zero)?
3) Why does dropout work? Explain the "ensemble" interpretation vs the "noise injection" interpretation.
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

2) Why does scaled dot-product attention divide by √d? Prove that this keeps the variance of the dot product ~1, and explain why softmax saturates without it.

3) Why do transformers use scaled dot-product attention instead of cosine similarity attention?

4) If bigger attention scores lead to sharper attention, why divide by √d at all? (Hint 1: think about gradient flow through softmax when inputs have high variance. Hint 2: consider training stability vs inference behavior.)

TODO: more questions here, loss funciton

# Contemporary Questions
TODO: add questions like "what does flash attention do differrently"


# Historical Questions

TODO: add questions like "why do we do this instead of this architechture, etc"

# Practice Problems

Pen-and-paper warmups before coding. Be able to work through these by hand.

## Forward/Backward Pass

1) Given a 2-layer MLP with ReLU, compute the forward pass:
   - Input: x = [1, 2]
   - W1 = [[0.5, -0.5], [0.5, 0.5]], b1 = [0, 0]
   - W2 = [[1, 1]], b2 = [0]
   - Activation: ReLU after layer 1
   - Compute: h = ReLU(W1 · x + b1), then y = W2 · h + b2

2) For that same network with target y* = 1 and MSE loss, apply the chain rule to compute ∂L/∂W2 and ∂L/∂W1.

3) Walk through one step of SGD with momentum:
   - Current weight: w = 0.5
   - Gradient: ∂L/∂w = 0.2
   - Velocity: v = 0.1
   - Learning rate: α = 0.1, momentum: β = 0.9
   - Compute: v_new = β·v + ∂L/∂w, then w_new = w - α·v_new

## Attention Mechanics

4) Manually compute the scaled dot-product attention output:
   - Q = [[1, 0], [0, 1]] (2 tokens, d=2)
   - K = [[1, 0], [0, 1]]
   - V = [[1, 2], [3, 4]]

5) Compute softmax([1, 2, 3]) by hand. Then compute softmax([10, 20, 30]). What happens and why?

## Architecture Analysis

6) Given a transformer config (L layers, H heads, d_model, d_ff, vocab V), calculate the total parameter count.

7) What is the memory and compute complexity of self-attention for sequence length N and dimension d?

## Debugging Scenarios

8) "Loss is NaN after a few training steps" - list the common causes and what you'd check.

9) "Model outputs nearly uniform attention weights across all positions" - what might cause this?

10) "Validation loss increases while training loss decreases" - what's happening and how do you address it?

# Coding Exercises

Be ready to implement from scratch and debug in real-time.

1) Implement a Multi-Head Attention block from scratch in PyTorch/NumPy without looking up documentation.

2) Manually derive and implement backpropagation for a simple 2-layer network.

3) Implement a KV cache for efficient autoregressive inference. Explain why it speeds up generation.

4) Implement positional encodings (sinusoidal and learned). What are the tradeoffs?

5) Implement beam search decoding.
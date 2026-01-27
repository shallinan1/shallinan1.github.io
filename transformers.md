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

    Key rule: $(AB)^T = B^T A^T$ — the transpose of a product reverses the order. Example: $(Xw)^T = w^T X^T$.

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

    A derivative is the slope of a function at a point: how much the output changes for a small change in input. Geometrically, it's the slope of the tangent line.

    A partial derivative is the same idea for functions of multiple variables: the slope with respect to one variable while holding the others fixed. For $f(x, y)$, the partial $\frac{\partial f}{\partial x}$ asks "if I nudge $x$ slightly, how much does $f$ change?" treating $y$ as constant.

    In neural networks, we compute partial derivatives of the loss with respect to each weight. Each partial tells us: if I nudge this weight, how much does the loss change?

    Note: The derivative itself is a *function*: it gives you the slope at any point you evaluate it. In training, we evaluate the gradient at the current weights, take a step, then re-evaluate at the new weights. That's why training is iterative: the gradient changes as we move through parameter space.

2) What is the gradient and how does it relate to the direction of steepest ascent? (gradient descent)

    For a function $f(x_1, x_2, \ldots, x_n)$, the gradient $\nabla f$ is a vector of all partial derivatives: $\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$.

    The gradient points in the direction of steepest *ascent*: the direction where $f$ increases fastest. Its magnitude tells you how steep that ascent is. To minimize a loss function $f$, we move in the opposite direction: $-\nabla f$. This is *gradient descent*.

    When parameters are organized in a matrix (like weights $W$), the gradient $\frac{\partial L}{\partial W}$ is also a matrix of the same shape. Each element $\frac{\partial L}{\partial W_{ij}}$ tells us how the loss changes when we nudge that specific weight. We still call this "the gradient" even though it's a matrix—it's the collection of all partial derivatives with respect to every parameter.

    Organizing both weights and gradients as matrices lets us compute and apply all updates in parallel - no loops over individual neurons. The gradient values are coupled (each depends on all the weights through the chain rule), but the matrix form lets us compute them in one shot with matrix operations.

    TODO more on gradient descent

3) What is the chain rule and why is it critical for neural networks?

    The chain rule tells us that for composed functions $f(g(x))$:
    $$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$$

    A neural network is a composition of functions. Each layer transforms the output of the previous layer, and the loss function sits on top. The chain rule lets us compute how the loss changes with respect to any weight by working backward through the layers. This is backpropagation.

    TODO derive chain rule

4) What is the gradient descent formula? What practical issues slow down training?

    The gradient descent update rule is $W_{t+1} = W_t - \alpha \frac{\partial L}{\partial W}$, where $\alpha$ is the learning rate.

    True local minima (all gradients zero) are rare in high-dimensional networks. The practical problems are:

    - **Flat regions**: small gradients mean slow progress
    - **Ill-conditioned curvature**: steep in some directions, shallow in others — one learning rate can't handle both

    TODO: Convexity — Why convex optimization is "easy" (one global minimum) and NNs are non-convex but work anyway.

5) What matrix calculus rules are essential for deriving gradients?

    **Transpose of a product:**
    $$(AB)^T = B^T A^T$$
    The transpose reverses the order. Example: $(Xw)^T = w^T X^T$.

    **Squared norm as dot product:**
    $$||v||^2 = v^T v$$
    The squared L2 norm is just the vector dotted with itself (sum of squared elements).

    **Derivative of linear term:**
    $$\frac{\partial}{\partial w}(a^T w) = a$$
    Same as scalar rule: $\frac{d}{dx}(ax) = a$

    **Derivative of quadratic term:**
    $$\frac{\partial}{\partial w}(w^T A w) = 2Aw \quad \text{(for symmetric } A \text{)}$$
    Same as scalar rule: $\frac{d}{dx}(ax^2) = 2ax$. The transpose in $w^T A w$ is just to make dimensions work.

    With these rules, you can derive the gradient for MSE loss:
    $$L = ||Xw - y||^2 = (Xw - y)^T(Xw - y) = w^T X^T X w - 2y^T X w + y^T y$$
    $$\frac{\partial L}{\partial w} = 2X^T X w - 2X^T y = 2X^T(Xw - y)$$

### Probability and Statistics

1) What is a probability distribution? What properties must it satisfy?

    A probability distribution describes how likely each outcome is. For discrete outcomes, it assigns a probability to each value. For continuous outcomes, it defines a density function where probability is the area under the curve.

    Two properties:

    - **Non-negative**: $P(x) \geq 0$ (discrete) or $p(x) \geq 0$ (continuous)
    - **Normalizes to 1**: $\sum_x P(x) = 1$ (discrete) or $\int p(x) dx = 1$ (continuous)

    **Discrete vs Continuous:**
    - Discrete: $P(x)$ is the probability itself, bounded between 0 and 1
    - Continuous: $p(x)$ is *density*, not probability. Density can exceed 1! (e.g., Uniform on $[0, 0.5]$ has density = 2). You integrate to get probability: $P(a < x < b) = \int_a^b p(x) dx$. The probability of any exact value is 0.

    TODO: Independence — $P(A,B) = P(A)P(B)$ for independent events. Why we can multiply likelihoods for i.i.d. data.

2) What is expected value? What is variance?

    **Expected value** is the probability-weighted average of all outcomes:
    $$\mathbb{E}[X] = \sum_x x \cdot P(x) \quad \text{(discrete)} \qquad \mathbb{E}[X] = \int x \cdot p(x) \, dx \quad \text{(continuous)}$$

    where $X$ is a random variable (can take different values) and $x$ is a specific value it can take. The mean is often written as $\mu = \mathbb{E}[X]$.

    **Variance** measures how spread out the distribution is:
    $$\text{Var}(X) = \sigma^2 = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - E[X]^2 = \mathbb{E}[X^2] - \mu^2$$

    It's the expected squared deviation from the mean. The second form comes from expanding $(X - \mu)^2$ and using linearity of expectation.

    **Standard deviation** $\sigma = \sqrt{\text{Var}(X)}$ has the same units as $X$.

3) Derive Bayes' theorem from the definition of conditional probability. What is ths significance?

    Recall given two conditional events $a$ and $b$, $P(a \cap b) = P(a | b) \times P(b)$. That is, we multiply the conditional probabilty by the probability of the event we're conditioning on. Conversely, we also know that $P(a \cap b) = P(b | a) \times P(a)$. Combining these together, we have:

    $$ P(a \cap b)= P(a|b) \times P(b) = P(b | a) \times P(a)$$

    As a result, we have both:
    $$P(a|b) = \frac{ P(b | a) \times P(a)}{P(b)} \ \text{ and } \ P(b|a) = \frac{P(a | b) \times P(b)}{P(a)}$$

    **Significance:** We often want $P(\text{hypothesis} | \text{data})$ — how likely is our model given what we observed ($D$)?

    $$\underbrace{P(H | D)}_{\text{posterior}} = \frac{\overbrace{P(D | H)}^{\text{likelihood}} \times \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(D)}_{\text{evidence}}}$$

    - **Prior** $P(H)$: belief before seeing data
    - **Likelihood** $P(D | H)$: how probable is this data if the hypothesis is true
    - **Posterior** $P(H | D)$: updated belief after seeing data
    - **Evidence** $P(D)$: total probability of data (normalizing constant)

    **Example:** A disease affects 1% of people. A test is 90% accurate (detects disease when present) but has a 5% false positive rate. You test positive — what's the probability you have the disease?

    - Prior: $P(\text{disease}) = 0.01$
    - Likelihood: $P(\text{positive} | \text{disease}) = 0.9$
    - False positive: $P(\text{positive} | \text{no disease}) = 0.05$
    - Evidence: $P(\text{positive}) = 0.9 \times 0.01 + 0.05 \times 0.99 = 0.059$

    $$P(\text{disease} | \text{positive}) = \frac{0.9 \times 0.01}{0.059} \approx 0.15$$

    Only 15%! The prior (disease is rare) dominates. This is why Bayes matters — intuition often ignores the base rate.

4) What is the maximum likelihood estimation (MLE) framework? Derive the MLE for a Gaussian distribution.

    MLE framework gives us a way to estimate parameters by finding the values that maximize the probability of observing our data. Given data $X$ and parameters $\theta$, we maximize:

    $$\hat{\theta} = \arg\max_\theta P(X | \theta)$$

    In practice, we maximize the log-likelihood (easier to work with, same result since log is monotonic):

    $$\hat{\theta} = \arg\max_\theta \log P(X | \theta)$$

    **Derivation for Gaussian:**

    Given $n$ i.i.d. samples $x_1, ..., x_n$ from $\mathcal{N}(\mu, \sigma^2)$, the likelihood is:

    $$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

    Log turns product into sum (using $\log(ab) = \log a + \log b$ and $\log e^x = x$):

    $$\log L = \sum_{i=1}^{n} \left[ \log\frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(x_i - \mu)^2}{2\sigma^2} \right]$$

    Note: $\log\frac{1}{\sqrt{2\pi\sigma^2}} = -\frac{1}{2}\log(2\pi\sigma^2) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2)$. Since this doesn't depend on $i$, summing $n$ times multiplies by $n$:

    $$\log L = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

    Take derivatives and set to zero:

    $$\frac{\partial}{\partial \mu}: \quad \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0 \quad \Rightarrow \quad \hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

    $$\frac{\partial}{\partial \sigma^2}: \quad -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i - \mu)^2 = 0 \quad \Rightarrow \quad \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2$$

    MLE finds the parameter values that jointly maximize the likelihood. Here we're finding the $(\mu, \sigma^2)$ pair that makes our data most probable. We set both partials to zero and solve the system. The solution: $\hat{\mu}$ is the sample mean, $\hat{\sigma}^2$ is the sample variance. Notice these are the familiar formulas you already use — MLE gives a principled justification for why they're the "right" estimates.

    TODO: move somewhere that an estimator $\hat{\theta}$ is unbiased if $\mathbb{E}[\hat{\theta}] = \theta$

    TODO: MAP vs MLE — Maximum a posteriori adds a prior to MLE. Connection to regularization (L2 regularization ↔ Gaussian prior).

5) Why do we maximize log-likelihood instead of likelihood directly?

    For i.i.d. data, likelihood is a product: $L(\theta) = \prod_{i=1}^{n} P(x_i | \theta)$. Log converts this to a sum: $\log L(\theta) = \sum_{i=1}^{n} \log P(x_i | \theta)$. This helps in two ways:

    **The value doesn't underflow.** A product of 1000 probabilities of 0.1 each is $0.1^{1000} = 10^{-1000}$, which underflows to 0. The log-likelihood is just $1000 \times \log(0.1) \approx -2303$.

    **Gradients don't vanish.** The gradient of a product (via product rule) is:
    $$\frac{\partial L}{\partial \theta} = \sum_{i=1}^{n} \left[ \frac{\partial P(x_i | \theta)}{\partial \theta} \cdot \prod_{j \neq i} P(x_j | \theta) \right]$$

    Each term multiplies by $n-1$ other probabilities, so it underflows too. With log-likelihood, the gradient is just a sum of independent terms — no vanishing products.

    **Why we can do this:** $\log$ is strictly monotonic, so $\arg\max_\theta L(\theta) = \arg\max_\theta \log L(\theta)$.

    **Note:** In deep learning, we minimize *negative* log-likelihood (NLL) since optimizers minimize by default.

6) How does the softmax function convert a list of raw logits into probabilities that sum to 1?

    The softmax function is $\text{softmax}(v_i) = \frac{e^{v_i}}{\sum_j e^{v_j}}$. It converts a vector into a discrete probability distribution.


    **Why it sums to 1:** The denominator is the sum of all numerators, so $\sum_i \text{softmax}(v_i) = \frac{\sum_i e^{v_i}}{\sum_j e^{v_j}} = 1$.

    **Why exp?** Why not just normalize raw values $v_i / \sum v_j$, or square them?
    - Raw values can be negative → can't be probabilities
    - Squaring breaks ordering: logits [-3, -1, 2] → squared [9, 1, 4] → most negative gets highest probability!
    - Exp is monotonic everywhere: larger logit → larger $e^{v_i}$ → larger probability. Always preserves ranking.
    - Exp amplifies differences: small gaps in logits become larger gaps in probabilities, making the distribution sharper/more confident.

7) What is entropy?

    **Entropy** (denoted $H$) measures the uncertainty in a probability distribution $p$:
    $$H(p) = -\sum_x p(x) \log p(x)$$

    where $p$ is a probability distribution over outcomes $x$, and $p(x)$ is the probability of outcome $x$.

    The $-\log p(x)$ term is the "surprise" of seeing outcome $x$ — rare events (small $p$) have high surprise (large $-\log p$). Entropy is the *expected* surprise.

    **Example:** $p = [0.7, 0.2, 0.1]$
    $$H = -(0.7 \log 0.7 + 0.2 \log 0.2 + 0.1 \log 0.1)$$

    - Certain distribution $[1, 0, 0]$ → entropy = 0 (no surprise)
    - Uniform distribution $[0.33, 0.33, 0.33]$ → maximum entropy (most uncertain)

    **Max entropy:** For $n$ outcomes, $H_{\max} = \log(n)$, achieved by uniform $p(x) = 1/n$. Making any outcome more likely reduces uncertainty.

8) What is cross-entropy? How does it relate to KL divergence?

    **Cross-entropy** between true distribution $p$ and predicted distribution $q$:
    $$H(p, q) = -\sum_x p(x) \log q(x)$$

    Compare to entropy: $H(p) = -\sum_x p(x) \log p(x)$. Cross-entropy uses $q$ inside the log instead of $p$.

    **Intuition:** Your surprise ($-\log q$) is based on what you predicted ($q$), but reality ($p$) determines how often each outcome happens.

    Example: Reality is $p = [0.9, 0.1]$, but you predict $q = [0.5, 0.5]$.
    - Outcome 1 happens 90% of the time, but you only assigned 50% → you're often more surprised than you should be
    - Cross-entropy is high because your predictions are bad

    If you predicted perfectly ($q = p$), cross-entropy equals entropy — the minimum possible surprise.

    **KL divergence** measures how different $q$ is from $p$:
    $$D_{KL}(p || q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

    Rearranging: $H(p, q) = H(p) + D_{KL}(p || q)$

    **In ML:** $p$ = true label (one-hot), $q$ = model's softmax output.
    - Entropy $H(p)$ of a one-hot is 0 (no uncertainty)
    - So minimizing cross-entropy $H(p, q)$ = minimizing KL divergence $D_{KL}(p || q)$
    - We're making $q$ as close to $p$ as possible

9) What is the relationship between minimizing cross-entropy and maximizing likelihood?

    **Cross-entropy with one-hot labels:** For true label $p$ (one-hot) and prediction $q$:
    $$H(p, q) = -\sum_x p(x) \log q(x) = -\log q(y_{true})$$

    where $y_{true}$ is the correct class. Only that class contributes (one-hot is 1 there, 0 elsewhere).

    **Likelihood** is the probability the model assigns to the correct class:
    $\mathcal{L}(\theta) = q(y_{true} | x; \theta)$. Taking the log, we get $\log \mathcal{L}(\theta) = \log q(y_{true} | x; \theta)$.

    **Connection:** Cross-entropy = negative log-likelihood:
    $H(p, q) = -\log q(y_{true}) = -\log \mathcal{L}(\theta)$

    So: **minimizing cross-entropy = minimizing negative log-likelihood = maximizing likelihood**.

    **For a dataset** with $N$ samples $(x_i, y_i)$:
    - Total cross-entropy loss: $\sum_{i=1}^{N} -\log q(y_i | x_i; \theta)$
    - Total log-likelihood: $\sum_{i=1}^{N} \log q(y_i | x_i; \theta)$

    They're negatives of each other. Minimizing total cross-entropy = maximizing total log-likelihood.

## ML Fundamentals

1) What is the difference between regression and classification?

    **Regression:** Predict a continuous value. Output is a real number.
    - Examples: house price, temperature, stock price
    - Loss: MSE (mean squared error) or MAE (mean absolute error) — penalize distance from true value

    **Classification:** Predict a discrete category. Output is a probability distribution over classes.
    - Examples: spam/not spam, digit 0-9, sentiment positive/negative
    - Loss: cross-entropy (measure how well predicted probabilities match true labels)

    The model architecture can be similar — the key difference is the output layer and loss function. Linear regression outputs a raw value; logistic regression adds a sigmoid to output a probability.

### Linear Regression

1) Derive the closed-form solution for linear regression using the normal equations.

    **Linear regression** models the target as a linear combination of features, $\hat{y} = Xw + b$, where $X$ is (N×D) input data (N samples, D features), $w$ is (D×1) weights, and $b$ is the bias/intercept.

    **Bias trick:** Append a column of 1s to $X$, making it (N×D+1). Now the bias is just another weight:
    $$\hat{y} = X'w' \quad \text{where } X' = [X | \mathbf{1}], \quad w' = [w; b]$$

    From here we drop the primes and assume $X$ includes the bias column.

    **Objective:** Given data $X$ (N×D) and targets $y$ (N×1), minimize MSE loss:
    $$L = ||Xw - y||^2$$

    **Take the gradient** (derived in Calculus Q5):
    $$\frac{\partial L}{\partial w} = 2X^T(Xw - y)$$

    **Set to zero and solve:**
    $$X^T(Xw - y) = 0$$
    $$X^T Xw = X^T y$$
    $$w = (X^T X)^{-1} X^T y$$


    **When does this fail?** $(X^TX)^{-1}$ doesn't exist when:
    - Columns of $X$ are linearly dependent (redundant features)
    - More features than samples ($D > N$)

    In these cases, use regularization (ridge regression adds $\lambda I$ to make it invertible) or gradient descent.

2) What is ridge regression? How does it modify the normal equation?

    Ridge regression adds L2 regularization to the loss:
    $$L = ||Xw - y||^2 + \lambda ||w||^2$$

    The regularization term $\lambda ||w||^2$ penalizes large weights, pushing them toward zero. (In practice, bias is not regularized — only the weights.)

    **Modified normal equation:**
    $$w = (X^T X + \lambda I)^{-1} X^T y$$

    Adding $\lambda I$ to $X^T X$ makes it always invertible (even when $D > N$ or columns are dependent). The diagonal entries become at least $\lambda$, so no zero eigenvalues.

    **Tradeoff:** Higher $\lambda$ = more regularization = smaller weights = simpler model, but may underfit. $\lambda = 0$ recovers ordinary least squares.

3) Why is MSE the "right" loss for regression?

    MSE is the MLE solution when you assume Gaussian noise on your targets. Minimizing squared error = maximizing likelihood under this assumption. If your errors aren't Gaussian (e.g., outliers, heavy tails), MSE isn't ideal — consider MAE or Huber loss instead. (See Appendix for full derivation.)

4) What is the bias-variance tradeoff? How does model complexity affect each?

    **Setup:** We assume observed targets are noisy versions of a true function:
    $$y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

    - $f(x)$ = the true underlying function (deterministic, what we want to learn)
    - $\epsilon$ = random noise (unpredictable, different for each observation)
    - $y$ = what we actually observe (true value + noise)
    - $\hat{y}$ = our model's prediction

    For MSE, expected error decomposes as:
    $$\mathbb{E}[(\hat{y} - y)^2] = \underbrace{(\mathbb{E}[\hat{y}] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{y} - \mathbb{E}[\hat{y}])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible}}$$

    where expectations are over different training sets.

    **Why is $\sigma^2$ irreducible?** Even if our model perfectly learned $f(x)$, we'd still have $\mathbb{E}[(\hat{y} - y)^2] = \mathbb{E}[(f(x) - (f(x) + \epsilon))^2] = \mathbb{E}[\epsilon^2] = \sigma^2$. The noise $\epsilon$ is random — no model can predict it.

    - **Bias:** How far is the average prediction from truth? (underfitting)
    - **Variance:** How much do predictions scatter around that average? (overfitting)
    - **Irreducible:** Noise in the data. No model can eliminate it.

    **Model complexity tradeoff:**
    - Simple model (e.g., linear): high bias, low variance
    - Complex model (e.g., deep network): low bias, high variance
    TODO this is handwavey

    Regularization (L2, dropout) reduces variance at the cost of slightly increased bias:
    - **Lower variance:** L2 pulls weights toward zero. Different training sets → models all pulled toward zero → more similar to each other.
    - **Higher bias:** If the true weights are large, L2 prevents the model from reaching them. Systematically biased toward zero.

    **For cross-entropy:** The irreducible error is $H(p)$, the entropy of the true distribution. With one-hot labels, $H(p) = 0$. With soft/noisy labels, you can't beat that entropy.

### Logistic Regression

1) Why can't we use linear regression for classification? Derive the logistic function as the solution.

    **Problem:** Linear regression outputs $\hat{y} = w^T x \in (-\infty, +\infty)$. For classification, we need probabilities in $[0, 1]$.

    **Solution:** Model the log-odds (logit) as linear:
    $$\log \frac{p}{1-p} = w^T x$$

    where $p = P(y=1|x)$. The log-odds can be any real number, so linear modeling makes sense here.

    **Solve for $p$:**
    $$\frac{p}{1-p} = e^{w^T x}$$
    $$p = (1-p) \cdot e^{w^T x}$$
    $$p = e^{w^T x} - p \cdot e^{w^T x}$$
    $$p(1 + e^{w^T x}) = e^{w^T x}$$
    $$p = \frac{e^{w^T x}}{1 + e^{w^T x}} = \frac{1}{1 + e^{-w^T x}}$$

    This is the **sigmoid function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$

    **Properties:** $\sigma(z) \in (0, 1)$, $\sigma(0) = 0.5$, $\sigma(-z) = 1 - \sigma(z)$

2) Derive the cross-entropy loss from the MLE of a Bernoulli distribution.

    For binary classification, $y \in \{0, 1\}$ follows a Bernoulli distribution:
    $$P(y|x) = p^y (1-p)^{1-y}$$

    where $p = P(y=1|x)$ is the model's prediction. (Check: if $y=1$, this gives $p$; if $y=0$, gives $1-p$.)

    **Likelihood** for $N$ samples:
    $$L = \prod_{i=1}^{N} p_i^{y_i} (1-p_i)^{1-y_i}$$

    **Log-likelihood:**
    $$\log L = \sum_{i=1}^{N} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]$$

    **Maximize log-likelihood = minimize negative log-likelihood:**
    $$-\log L = -\sum_{i=1}^{N} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]$$

    This is exactly **binary cross-entropy loss**. Just like MSE is MLE for Gaussian noise, cross-entropy is MLE for Bernoulli labels.

3) How does logistic regression extend to multiclass classification?

    **Binary:** One weight vector. $z = w^T x$ gives a single logit, sigmoid converts to probability.

    **Multiclass (K classes):** One weight vector per class. Each class $c$ has its own linear function:
    $$z_c = w_c^T x$$

    This gives $K$ logits (one per class). Softmax converts them to probabilities:
    $$p_c = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}}$$

    **Loss:** Categorical cross-entropy: $-\sum_c y_c \log p_c$. With one-hot labels ($y_c = 1$ for correct class, 0 otherwise), this simplifies to $-\log p_{\text{true}}$. This is called **softmax regression** or **multinomial logistic regression**. Still a linear model (decision boundaries are hyperplanes).

4) Why is there no closed-form solution for logistic regression?

    **Linear regression:** $\frac{\partial L}{\partial w} = X^T(Xw - y) = 0$. Linear in $w$, so we can solve algebraically: $w = (X^T X)^{-1} X^T y$.

    **Logistic regression:** Loss is $L = -\sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)]$ where $p_i = \sigma(w^T x_i)$ (bias absorbed into $w$).

    $$\frac{\partial L}{\partial w} = \sum_i (p_i - y_i) \cdot x_i = 0$$

    Can't isolate $w$ because $p_i = \sigma(w^T x_i)$ is nonlinear in $w$.

    **Solution:** Iterative optimization.
    - Gradient descent: $w \leftarrow w - \alpha \nabla L$
    - Newton's method: $w \leftarrow w - H^{-1} \nabla L$ (uses Hessian, converges faster)

    Both work well because cross-entropy loss is **convex** for logistic regression — no local minima, guaranteed to find global optimum.

5) Why is the decision boundary linear even though logistic regression uses a nonlinear sigmoid?

    We predict class 1 when $P(y=1|x) > 0.5$: $\sigma(w^T x) > 0.5$. Since $\sigma(0) = 0.5$, this simplifies to: $w^T x > 0$. This is a linear equation in $x$; the boundary is a hyperplane. The sigmoid squashes outputs to probabilities but doesn't bend the boundary.

    **For nonlinear boundaries**, you need: Feature engineering (add $x^2$, $x_1 x_2$, etc.), Kernel methods, Neural networks (multiple layers with nonlinearities)

    **Classic example — XOR:** Points (0,0), (1,1) are class 0; points (0,1), (1,0) are class 1. No straight line can separate them. Logistic regression fails. A neural network with one hidden layer can solve it.

### Neural Networks

1) What is a fully connected (dense) layer? How does it transform its input?

    A fully connected (dense) layer computes a linear transformation followed by an optional nonlinearity:
    $$y = \sigma(Wx + b)$$

    where $x$ is the input (D×1), $W$ is the weight matrix (M×D), $b$ is the bias (M×1), and $\sigma$ is an activation function.

    **"Fully connected" means:** Every output neuron connects to every input. The weight $W_{ij}$ determines how much input $j$ contributes to output $i$. This is in contrast to convolutional layers (local connections) or attention (dynamic connections).

    **Transformation:** Maps D-dimensional input to M-dimensional output. Each of the M output neurons computes a dot product between its weight vector (one row of $W$) and the input, then adds bias. As discussed in the matrix multiplication section, each neuron's weights encode a "pattern" — the dot product measures how much the input matches that pattern.

    **Batched form:** For N samples, $X$ is (N×D), and $Y = XW^T + b$ gives (N×M) outputs. All samples processed in parallel.

2) Why are nonlinear activation functions (ReLU, GELU) necessary? What happens with only linear layers?

    **Without nonlinearities, depth is useless.** Stacking linear layers collapses to a single linear layer:
    $$y = W_2(W_1 x) = (W_2 W_1)x = W'x$$

    No matter how many layers, the network can only learn linear functions. Linear functions can't solve XOR, can't learn curves, can't approximate complex decision boundaries.

    **Nonlinearities break this collapse.** With $y = W_2 \cdot \text{ReLU}(W_1 x)$, the composition is no longer reducible. Each layer can learn features that the next layer builds on. This is what makes deep networks expressive.

    **Common activations:**
    - **ReLU:** $\max(0, x)$. Simple, sparse (zeros out negatives), fast. Problem: "dead neurons" — if a neuron's input is always negative, gradient is always 0, it never updates.
    - **Leaky ReLU:** $\max(0.01x, x)$. Small slope for negatives prevents dead neurons.
    - **GELU:** $x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF. Smooth approximation to ReLU, used in transformers (BERT, GPT). Slightly better empirically; the smoothness may help optimization.
    - **SiLU/Swish:** $x \cdot \sigma(x)$. Similar to GELU, used in newer models.

    **Why ReLU works despite being so simple:** It's piecewise linear, so gradients don't vanish (unlike sigmoid/tanh which saturate). The "kink" at 0 provides the nonlinearity. Networks with ReLU are universal approximators — they can approximate any continuous function by combining enough piecewise linear pieces.

3) Derive backpropagation for a simple 2-layer network. What is the computational complexity?

    **Setup:** Input $x$ (1×D) → $z_1 = xW_1$ (D×H) → $h = \text{ReLU}(z_1)$ → $\hat{y} = hW_2$ (H×1) → $L = (\hat{y} - y)^2$

    Forward pass computes and stores $z_1, h, \hat{y}, L$. Backward pass applies chain rule from output to input:
    1. $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$ — gradient of MSE
    2. $\frac{\partial L}{\partial W_2} = h^T \cdot \frac{\partial L}{\partial \hat{y}}$ — shape (H×1), matches $W_2$
    3. $\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} \cdot W_2^T$ — shape (1×H), note the transpose to "fan in"
    4. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \odot \mathbf{1}_{z_1 > 0}$ — ReLU gates the gradient (element-wise)
    5. $\frac{\partial L}{\partial W_1} = x^T \cdot \frac{\partial L}{\partial z_1}$ — shape (D×H), matches $W_1$

    **Pattern:** Weight gradient = (layer input)$^T$ × (upstream gradient). Gradient to pass back = (upstream gradient) × (weights)$^T$.

    **Complexity:** Forward pass does one matmul per layer, $O(DH)$ for (D×H) weights. Backward does two matmuls per layer (weight gradient + propagate back). Total: $O(\sum_{\text{layers}} n_{in} \times n_{out})$ — linear in parameters. Backprop costs ~2-3× forward pass.

4) In the backward pass, why do we multiply by $W^T$ instead of $W$? (Hint: think about "fanning out" vs "fanning in")

    Concrete example in backpropagation:
    - **Forward pass**: $Y = XW$ where $X$ is $(N \times D)$ and $W$ is $(D \times M)$
    - **Backward pass**: you receive gradient $\frac{\partial L}{\partial Y}$ with shape $(N \times M)$ and need gradient $\frac{\partial L}{\partial X}$ with shape $(N \times D)$
    - To get there: $(N \times M) \cdot (M \times D) = (N \times D)$, so you need $W^T$

    The weights "fan out" from $D$ inputs to $M$ neurons in the forward pass. To propagate gradients backward, you "fan in" from $M$ neurons back to $D$ inputs. Same weights, opposite direction.
    
5) What causes the vanishing/exploding gradient problem? How do ReLU and residual connections help?

    Backpropagation multiplies gradients through layers via the chain rule. For a deep network with $L$ layers:
    $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$
    Each $\frac{\partial h_{l+1}}{\partial h_l}$ involves the activation's derivative and the weight matrix. If these terms are consistently $< 1$, gradients shrink exponentially. If consistently $> 1$, they explode.

    Gradients vanish with sigmoid and tanh because these activations saturate for large inputs — their derivatives approach 0. After many layers, gradients become negligibly small, early layers barely update, and the network can't learn long-range dependencies. Gradients explode when weight matrices have large singular values, causing exponential growth. Training becomes unstable — loss spikes or goes to NaN.

    ReLU helps by *not* shrinking gradients. Sigmoid's derivative is $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, which is maximized when $\sigma(x) = 0.5$, giving $0.5 \times 0.5 = 0.25$. So each layer multiplies gradients by at most 0.25 — after 10 layers: $0.25^{10} \approx 10^{-6}$. ReLU's derivative is exactly 1 for $x > 0$, so gradients pass through unchanged. It's not that ReLU boosts anything — it just stops the bleeding. For $x < 0$ the gradient is 0, which can cause "dead neurons," but that's a different problem than vanishing gradients (0 vs exponentially small).

    Residual connections provide a "highway" for gradients to skip problematic layers entirely (see Q7 for details). Together, ReLU prevents per-layer shrinking and residuals prevent cross-layer shrinking, enabling very deep networks.

6) Why does weight initialization matter? What do Xavier and He initialization do?

    Bad initialization causes vanishing/exploding gradients before training even starts. If weights are too small, activations shrink layer by layer; too large, they explode.

    **Xavier (Glorot) initialization:** $W \sim \mathcal{N}(0, \frac{1}{n_{in}})$ or $\mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$. Designed for tanh/sigmoid — keeps variance ~1 through layers.

    **He initialization:** $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$. Designed for ReLU — accounts for ReLU killing half the activations (hence the 2×).

    The key insight: variance of layer output depends on (variance of weights) × $n_{in}$. Each neuron sums $n_{in}$ terms, and variance of a sum scales with the number of terms. These schemes set weight variance to $\frac{1}{n_{in}}$ (or $\frac{2}{n_{in}}$) to counteract this, keeping activations stable.

7) What are residual (skip) connections and how do they help gradients flow through deep networks?

    A residual block computes $y = F(x) + x$ instead of $y = F(x)$, where $x$ is the block's input (activations from the previous layer) and $F$ is typically a few layers (conv-batchnorm-relu or similar). The input $x$ "skips" past $F$ and gets added to the output.

    Why this helps gradients: during backprop, each layer computes how its output changes w.r.t. its input so the gradient can propagate to earlier layers. For a residual block, $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$. Even if $\frac{\partial F}{\partial x}$ vanishes, the $+I$ term lets gradients flow through unchanged. This "highway" lets gradients reach early layers without shrinking, which is why ResNets can train 100+ layers while plain networks struggle past 20.

    When $F(x)$ and $x$ have different dimensions, the skip connection uses a learned projection: $y = F(x) + W_s x$, where $W_s$ matches dimensions (typically a 1×1 convolution in CNNs or a linear layer in transformers).

8) What is the universal approximation theorem and what are its practical limitations?

#### Graph Neural Networks

1) What is the core idea of GNNs? What is the over-smoothing problem in deep GNNs?

    GNNs learn node representations by aggregating information from neighbors. Each layer updates a node's embedding by combining its current embedding with a summary (sum, mean, max) of its neighbors' embeddings. This is "message passing."

    **Over-smoothing:** As you stack more layers, each node's representation incorporates information from increasingly distant neighbors. After many layers, all nodes converge to similar representations — they've "seen" the whole graph and lost local structure. This limits GNN depth (typically 2-4 layers work best).

### Optimization Algorithms

1) Why does SGD work despite using noisy gradient estimates? What role does the noise play?

The noise averages out to 0 in expectation since their values are small. The noise helps get us out of local minima or reigons with near-zero gradients.
2) Derive the momentum update rule. Why does it help with saddle points and narrow valleys?
3) Explain Adam: how does it combine momentum with adaptive learning rates? When does AdamW help?
4) What is gradient clipping and when is it used?

    Gradient clipping caps gradient magnitudes to prevent exploding gradients from destabilizing training.

    **Clip by value:** Cap each gradient element to $[-c, c]$. Simple but can change gradient direction.

    **Clip by norm (more common):** If $||\nabla L|| > c$, scale the entire gradient: $\nabla L \leftarrow c \cdot \frac{\nabla L}{||\nabla L||}$. Preserves direction, just limits step size.

    Used heavily in RNNs/LSTMs (long unrolls cause gradient explosion) and transformers. Typical values: 1.0 for transformers, 5.0 for RNNs.

5) What is the learning rate warmup and why is it important for transformers?

6) What happens if your learning rate is too large? Too small?

    **Too large:**
    - Loss oscillates wildly or diverges to infinity/NaN
    - Overshooting minima — each step jumps past the optimal point
    - Gradients explode as you land in bad regions of loss landscape

    **Too small:**
    - Training is very slow
    - May get stuck in sharp local minima or saddle points
    - Takes forever to converge (or never does in practice)

    **Diagnosis:** Plot loss curve. Oscillating/exploding = too large. Barely decreasing = too small. Good learning rate shows steady decrease with occasional bumps.

### Regularization

1) Derive why L2 regularization is equivalent to a Gaussian prior on weights (MAP estimation).
2) What's the difference between L1 and L2 regularization? Why does L1 promote sparsity (drive weights to exactly zero)?
3) Why does dropout work? Explain the "ensemble" interpretation vs the "noise injection" interpretation.
4) Derive the batch normalization forward pass. Why does it help with internal covariate shift?
5) Why do transformers use layer norm instead of batch norm? Where is it placed (pre-LN vs post-LN)?

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

    **Autoregressive factorization:** Any sequence probability can be decomposed as:
    $$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, \ldots, x_{t-1})$$

    Each token's probability is conditioned on all previous tokens. Language models learn to predict $P(x_t | x_{<t})$.

    **Connection to cross-entropy:** At each position, the model outputs a distribution $q$ over vocabulary, and the true next token is $p$ (one-hot). Cross-entropy loss:
    $$L = -\sum_t \log q(x_t | x_{<t})$$

    This is exactly **negative log-likelihood** of the sequence. Minimizing cross-entropy = maximizing the probability the model assigns to the correct next token at every position.

    **Training objective:** See the correct history $x_{<t}$, predict next token $x_t$. Summed over all positions in all training sequences.

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
3) What is temperature in softmax sampling? What are top-k and top-p (nucleus) sampling?

    Temperature scales the logits before softmax: $\text{softmax}(v_i / T)$
    - $T < 1$: sharper distribution, more confident, more repetitive
    - $T > 1$: flatter distribution, more random, more creative
    - $T = 1$: original distribution

    **Top-k**: Only sample from the k most probable tokens (zero out the rest).

    **Top-p (nucleus)**: Only sample from the smallest set of tokens whose cumulative probability ≥ p. Adapts to the distribution — keeps more tokens when uncertain, fewer when confident.

4) What is the length bias problem in seq2seq? How is it typically addressed?

### Attention (Pre-Transformer)

1) If we can't process words left-to-right, how else could we determine which words are related to each other?
2) What does it mean to "attend" to specific parts of an input sequence while generating an output?
3) Derive Bahdanau (additive) attention. What are the query, key, and value in this formulation?
4) What is Luong (multiplicative) attention and how does it differ?
5) How does attention solve the information bottleneck? What new bottleneck does it create (hint: complexity)?
6) If we process all words in parallel (not sequentially), how can we encode their position or order?



# Basic Transformer Questions (High Level)

1) How does convolution decide how much weight to give each input position? How does attention?

    **Convolution:** Uses fixed, learned weights. A kernel of size $k$ learns $k$ weights at training time, and these same weights are applied identically at every position. The weights don't depend on what the input actually contains — position 2 always gets weight $w_2$, regardless of the content there.

    **Attention:** Computes weights dynamically based on content. Each position's weight comes from a query-key dot product: "how relevant is this position to what I'm looking for?" The same input at a different position (or with different surrounding context) can receive completely different attention weights.

    **Key distinction:** Convolution is *content-agnostic* (weights fixed by position), attention is *content-dependent* (weights computed from the data).

2) What if you set the convolution kernel size equal to the sequence length?

    Then every output position can "see" the entire input — same receptive field as attention. But the weights are still fixed and position-based. Position 0 always gets $w_0$, position 1 always gets $w_1$, etc.

    This means:
    - Can't handle variable-length sequences (kernel size is fixed)
    - Can't dynamically focus on relevant positions (weights don't depend on content)
    - Parameter count scales linearly with sequence length

    Attention solves all three: works on any length, focuses where content is relevant, and has fixed parameter count regardless of sequence length. The tradeoff is $O(N^2)$ compute to compute all pairwise similarities.

3) What is the high-level idea of how transformers work?

   <details><summary>Answer</summary>
    Test
   </details>

4) Why does scaled dot-product attention divide by √d? Prove that this keeps the variance of the dot product ~1, and explain why softmax saturates without it.

5) Why do transformers use scaled dot-product attention instead of cosine similarity attention?

6) If bigger attention scores lead to sharper attention, why divide by √d at all? (Hint 1: think about gradient flow through softmax when inputs have high variance. Hint 2: consider training stability vs inference behavior.)

TODO: more questions here, loss funciton

# Contemporary Questions

## KV Cache

1) What is the KV cache and why is it essential for efficient autoregressive generation?

    During autoregressive generation, we produce one token at a time. At step $t$, we need to compute attention over all previous tokens $1, 2, ..., t-1$ plus the new token $t$.

    Without KV cache: Recompute K and V for *all* tokens at every step.
    - Step 1: compute K, V for token 1
    - Step 2: compute K, V for tokens 1, 2
    - Step 3: compute K, V for tokens 1, 2, 3
    - Total: $O(n^2)$ K/V computations for n tokens

    With KV cache: Store K and V from previous steps, only compute for new token.
    - Step 1: compute K₁, V₁, store them
    - Step 2: compute K₂, V₂, store them, reuse K₁, V₁
    - Step 3: compute K₃, V₃, store them, reuse K₁, V₁, K₂, V₂
    - Total: $O(n)$ K/V computations

    Why it works: In causal attention, token $t$ only attends to tokens $\leq t$. The K and V vectors for past tokens don't change — they only depend on those tokens' positions, not future tokens. So we can cache them.

    Q is not cached because we only need the query for the *current* token (we're computing attention *from* the new token *to* all previous tokens).

2) What is the memory cost of the KV cache? When does it become a bottleneck?

    For each layer, we store K and V matrices of shape (batch_size, num_heads, seq_len, head_dim). Memory per layer: $2 \times B \times H \times L \times d_h \times \text{bytes\_per\_param}$, where $B$ = batch size, $H$ = num heads, $L$ = sequence length, $d_h$ = head dimension.

    Concrete example (Llama-2 70B): 80 layers, 64 heads, head_dim = 128, fp16 (2 bytes). Per token: $2 \times 80 \times 64 \times 128 \times 2 = 2.6$ MB. For 4096 context: ~10.5 GB just for KV cache.

    Bottlenecks: long contexts (KV cache grows linearly with sequence length), large batch sizes (multiplies cache size), memory-constrained deployment. This is why Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and sliding window attention exist.

3) How do Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce KV cache size?

    Standard Multi-Head Attention (MHA): Each head has its own K, V projections. KV cache size: $2 \times H \times d_h$ per token per layer.

    Multi-Query Attention (MQA): All heads share a *single* K and V. KV cache size: $2 \times d_h$ per token per layer (H× smaller). Tradeoff: some quality degradation since heads can't specialize their K/V.

    Grouped-Query Attention (GQA): Compromise — groups of heads share K/V. If $G$ groups: KV cache size is $2 \times G \times d_h$ (between MHA and MQA). Example: 8 groups with 64 heads = 8 heads share each K/V. Used in Llama-2 70B, Mistral, etc.

    Why this works: Empirically, K/V representations are more similar across heads than Q representations. Sharing K/V hurts less than sharing Q would.

## Flash Attention

1) What problem does Flash Attention solve? Why is standard attention memory-inefficient?

    Standard attention: `S = Q @ K.T` (N×N), `P = softmax(S)` (N×N), `O = P @ V` (N×d). The problem: we materialize the full $N \times N$ attention matrix in GPU memory (HBM). Memory is $O(N^2)$ — for N=8192, d=128: attention matrix is 256MB (fp32), but Q/K/V are only 4MB each.

    The real bottleneck is memory bandwidth, not compute. Modern GPUs have massive compute (TFLOPS) but limited memory bandwidth. Reading/writing that huge attention matrix to HBM is slow.

    Flash Attention insight: Never materialize the full N×N matrix. Compute attention in *tiles* that fit in fast SRAM, write only the final output to HBM.

2) How does Flash Attention use tiling to reduce memory I/O?

    GPU memory hierarchy: HBM (High Bandwidth Memory) is large (~40GB) but slow (~2TB/s). SRAM (on-chip) is small (~20MB) but fast (~19TB/s).

    Flash Attention algorithm: (1) Divide Q, K, V into blocks that fit in SRAM. (2) For each block of Q, load it into SRAM. (3) For each block of K, V: load into SRAM, compute local attention scores, compute local softmax with running max for numerical stability, accumulate weighted V contributions. (4) Write final output block to HBM.

    Softmax is tricky because it needs the max over the entire row. Flash Attention uses the "online softmax" trick — track running statistics and rescale as you process each K/V block. Memory: $O(N)$ instead of $O(N^2)$.

3) What are the practical speedups from Flash Attention?

    Memory reduction from $O(N^2)$ to $O(N)$ enables much longer contexts. Wall-clock speedup of 2-4× on typical workloads, despite doing the "same" computation — the speedup comes from reduced memory I/O, not reduced FLOPs. Standard attention runs out of memory around N=8K-16K on typical GPUs; Flash Attention can handle N=64K+. Unlike sparse or linear attention approximations, Flash Attention computes *exact* attention.

4) What is Flash Attention 2 and what does it improve?

    Flash Attention 2 optimizes parallelization. FA1 parallelizes over batch size and heads — each thread block handles one (batch, head) pair, iterates over sequence. FA2 additionally parallelizes over sequence length (both Q and K/V dimensions), has better work partitioning between warps, and reduces non-matmul FLOPs (softmax bookkeeping). Result: ~2× speedup over FA1, reaching 50-73% of theoretical max FLOPS on A100 (vs ~25-40% for FA1).


# Historical Questions

TODO: add questions like "why do we do this instead of this architechture, etc"

# Practice Problems

Pen-and-paper warmups before coding. Be able to work through these by hand.

## Forward/Backward Pass

1) Walk through 3 iterations of gradient descent for $\hat{y} = wx$ with $w_0 = 5$, $x = 2$, $y = 2$, $\alpha = 0.1$, and MSE loss $L = (\hat{y} - y)^2$.

    **Iteration 1:**

    *Forward:*
    - $\hat{y} = 5 \cdot 2 = 10$
    - $L = (10 - 2)^2 = 64$

    *Backward (chain rule):*
    - $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = 2(8) = 16$
    - $\frac{\partial \hat{y}}{\partial w} = x = 2$
    - $\frac{\partial L}{\partial w} = 16 \cdot 2 = 32$

    *Update:*
    - $w \leftarrow 5 - 0.1 \cdot 32 = 1.8$

    **Iteration 2:**

    *Forward:*
    - $\hat{y} = 1.8 \cdot 2 = 3.6$
    - $L = (3.6 - 2)^2 = 2.56$

    *Backward:*
    - $\frac{\partial L}{\partial w} = 2(3.6 - 2) \cdot 2 = 6.4$

    *Update:*
    - $w \leftarrow 1.8 - 0.1 \cdot 6.4 = 1.16$

    **Iteration 3:**

    *Forward:*
    - $\hat{y} = 1.16 \cdot 2 = 2.32$
    - $L = (2.32 - 2)^2 = 0.1024$

    *Backward:*
    - $\frac{\partial L}{\partial w} = 2(0.32) \cdot 2 = 1.28$

    *Update:*
    - $w \leftarrow 1.16 - 0.1 \cdot 1.28 = 1.032$

    Converging toward $w = 1$ (optimal since $y/x = 2/2 = 1$).

2) Given a 2-layer MLP with ReLU, compute the forward pass:
   - Input: x = [1, 2]
   - W1 = [[0.5, -0.5], [0.5, 0.5]], b1 = [0, 0]
   - W2 = [[1, 1]], b2 = [0]
   - Activation: ReLU after layer 1
   - Compute: h = ReLU(W1 · x + b1), then y = W2 · h + b2

3) For that same network with target y* = 1 and MSE loss, apply the chain rule to compute ∂L/∂W2 and ∂L/∂W1.

4) Walk through one step of SGD with momentum:
   - Current weight: w = 0.5
   - Gradient: ∂L/∂w = 0.2
   - Velocity: v = 0.1
   - Learning rate: α = 0.1, momentum: β = 0.9
   - Compute: v_new = β·v + ∂L/∂w, then w_new = w - α·v_new

5) Derive the gradient of layer normalization with respect to its input.

    **Layer norm forward pass** (for a single sample with $d$ features):
    $$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2, \quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

    **Why is this tricky?** Each output $y_i$ depends on $x_i$ directly, but also on $\mu$ and $\sigma^2$, which depend on *all* $x_j$. We need to account for both paths.

    **Derivation:** Let $\frac{\partial L}{\partial y_i}$ be the upstream gradient. We want $\frac{\partial L}{\partial x_i}$.

    First, compute intermediate gradients:
    $$\frac{\partial L}{\partial \hat{x}_i} = \gamma \cdot \frac{\partial L}{\partial y_i}$$

    $$\frac{\partial L}{\partial \sigma^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2}$$

    $$\frac{\partial L}{\partial \mu} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{-2}{d}\sum_i(x_i - \mu)$$

    Note: The second term in $\frac{\partial L}{\partial \mu}$ is 0 because $\sum_i(x_i - \mu) = 0$ by definition of $\mu$.

    Finally, combining all three paths ($x_i$ affects $\hat{x}_i$ directly, and through $\mu$ and $\sigma^2$):
    $$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{d} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{d}$$

    **Simplified form:** Let $\sigma = \sqrt{\sigma^2 + \epsilon}$ and $g_i = \frac{\partial L}{\partial \hat{x}_i}$:
    $$\frac{\partial L}{\partial x_i} = \frac{1}{d \cdot \sigma}\left(d \cdot g_i - \sum_j g_j - \hat{x}_i \sum_j g_j \hat{x}_j\right)$$

    **Intuition:** The gradient has three components: (1) the direct gradient scaled by $1/\sigma$, (2) a mean-centering term (subtracts mean of gradients), (3) a variance-stabilizing term (removes correlation with normalized values).

## Attention Mechanics

6) Manually compute the scaled dot-product attention output:
   - Q = [[1, 0], [0, 1]] (2 tokens, d=2)
   - K = [[1, 0], [0, 1]]
   - V = [[1, 2], [3, 4]]

7) Compute softmax([1, 2, 3]) by hand. Then compute softmax([10, 20, 30]). What happens and why?

## Architecture Analysis

8) Given a transformer config (L layers, H heads, d_model, d_ff, vocab V), calculate the total parameter count.

9) What is the memory and compute complexity of self-attention for sequence length N and dimension d?

## Debugging Scenarios

10) Why do we overfit a tiny batch when debugging training?

    Before training on full data, overfit 1-2 batches first. If the model can't memorize a tiny batch, something is broken.

    **What it tests:**
    - Forward pass computes correctly
    - Backward pass updates weights
    - Loss can decrease at all
    - No bugs in data loading/preprocessing

    **Expected:** Loss drops to ~0 quickly (model memorizes the batch). If not, debug before scaling up.

    **Common failures:** Wrong loss function, frozen weights, learning rate = 0, bad data pipeline, labels don't match inputs.

11) "Loss is NaN after a few training steps" - list the common causes and what you'd check.

12) "Model outputs nearly uniform attention weights across all positions" - what might cause this?

13) "Validation loss increases while training loss decreases" - what's happening and how do you address it?

## Long-Horizon Planning & Reasoning

1) What is the ReAct pattern for LLM agents?

    **ReAct (Reason + Act):** Interleave reasoning and actions in a loop:
    ```
    Thought: I need to find the population of Tokyo
    Action: search("Tokyo population")
    Observation: Tokyo has 13.96 million people
    Thought: Now I need to compare to New York...
    Action: search("New York population")
    ...
    ```

    **Why it works:**
    - Reasoning traces help the model plan next action
    - Observations ground the model in real information (not hallucination)
    - Interleaving prevents long chains of unsupported reasoning

    **Components:** LLM (generates thoughts + actions), Tools (execute actions, return observations), Loop (until task complete or max steps)

2) What is chain-of-thought prompting and its variants?

    **Chain-of-thought (CoT):** Generate intermediate reasoning steps before the final answer.

    **Why it helps:**
    - Breaks complex problem into simpler subproblems
    - Each step conditions on previous steps
    - More "compute" via output tokens
    - Interpretable/debuggable

    **Variants:**
    - **Zero-shot CoT:** Append "Let's think step by step" to prompt
    - **Few-shot CoT:** Provide examples with reasoning traces
    - **Self-consistency:** Sample N reasoning paths, majority vote on final answer
    - **Tree-of-thought:** Branch into multiple reasoning paths, evaluate each (like MCTS but simpler)

    **Limitation:** Still left-to-right, no backtracking. For complex problems, use search (MCTS) or self-consistency.

3) How do LLM agents use tools and function calling?

    **Problem:** LLMs can't do math reliably, don't have real-time info, can't take actions in the world.

    **Solution:** Give LLM access to tools. Model outputs a structured function call, system executes it, result goes back to model.

    ```
    User: What's 17.3 * 284.9?
    LLM: <function_call>calculator(17.3 * 284.9)</function_call>
    System: 4930.77
    LLM: The result is 4930.77
    ```

    **Common tools:** Calculator, web search, code interpreter, APIs, databases

    **How it's trained:**
    - Fine-tune on examples of (query, function_call, result, response)
    - Or few-shot prompting with tool use examples

    **Key design choices:**
    - Tool descriptions in system prompt
    - Structured output format (JSON, XML)
    - When to call tools vs. answer directly

4) How do you handle long-horizon tasks that exceed context length?

    **Problem:** Agent needs to complete a task over many steps. Context fills up with observations, previous actions, etc.

    **Approaches:**

    **Summarization:** Periodically summarize history, keep summary instead of full trace.

    **Memory systems:**
    - Short-term: Recent context (fits in window)
    - Long-term: Vector DB of past observations/facts, retrieve relevant ones

    **Hierarchical planning:**
    - High-level plan: ["research topic", "write outline", "draft sections", "revise"]
    - Execute each step with fresh context
    - Only carry forward the outputs, not full traces

    **Scratchpad/state:** Maintain structured state (key facts, current plan, completed steps) instead of raw history.

5) What is task decomposition and replanning for LLM agents?

    **Decomposition:** Break complex task into subtasks.
    ```
    Task: "Book travel to NeurIPS"
    Subtasks:
    1. Find conference dates and location
    2. Search flights
    3. Search hotels near venue
    4. Compare options
    5. Book best combination
    ```

    **Why it helps:**
    - Each subtask is tractable
    - Can verify intermediate results
    - Parallelizable (search flights and hotels simultaneously)
    - Error recovery (if step 2 fails, retry just that step)

    **Replanning:** If execution fails or new info arrives, regenerate the plan.
    - "No direct flights" → replan with layover options
    - Dynamic, not fixed plan

    **Examples:** HuggingGPT, Plan-and-Solve, AutoGPT-style agents

6) What does it mean to use an LLM as a "world model"?

    **World model:** A model that predicts how the world evolves given actions. Traditional world models (in RL) predict next state: $s_{t+1} = f(s_t, a_t)$.

    **LLM as world model:** The LLM itself simulates what happens next. Instead of taking actions in the real world, you "imagine" outcomes by generating text.

    **Where this shows up:**

    **Chain-of-thought as simulation:** Each reasoning step is a "mental action." The LLM predicts what follows from that step — effectively simulating a reasoning trajectory through problem space.

    **Self-consistency:** Sample N reasoning paths from the LLM. Each path is a different "rollout" through the world model. Majority vote aggregates these simulated trajectories.

    **Tree-of-thought:** Branch at decision points, explore multiple futures. The LLM generates possible continuations (world model predicts outcomes), then evaluates which branches look promising.

    **Planning with lookahead:** Before taking a real action, simulate "if I do X, then Y will happen." LLM generates the hypothetical Y. Compare simulated outcomes of different actions, pick best.

    **Example:**
    ```
    Task: "Should I send this email?"

    Simulate action A (send now):
    → "Recipient sees it at 11pm, might seem urgent..."
    → "They reply tomorrow morning..."

    Simulate action B (wait until morning):
    → "Arrives during work hours..."
    → "Seems more professional..."

    Compare simulated outcomes → choose B
    ```

    **Limitations:**
    - LLM world models hallucinate — simulated outcomes may not match reality
    - No grounding unless you actually execute actions and observe
    - Works best for reasoning (abstract) vs. physical prediction (needs real physics)

    **Key insight:** Chain-of-thought, self-consistency, and tree-of-thought are all using the LLM as an implicit world model. The recent framing makes this explicit and connects to RL planning literature.

## Alignment / RLHF

1) What is MCTS and how does it apply to LLM reasoning?

    **Monte Carlo Tree Search (MCTS)** is a search algorithm that explores possible actions by building a tree and evaluating paths without exhaustively completing all of them.

    **Standard LLM generation:** Greedy or sampling-based, left-to-right. Once you generate tokens, you're committed — no backtracking.

    **MCTS for LLMs:** Explore multiple reasoning paths:
    - Each node = partial response (a reasoning step, sentence, or paragraph)
    - **Select** = pick which node to expand using UCB (Upper Confidence Bound):
      $$UCB = \frac{\text{value}}{\text{visits}} + c \cdot \sqrt{\frac{\ln(\text{parent visits})}{\text{visits}}}$$
      First term exploits high-value nodes; second term explores undervisited nodes.
    - **Expand** = generate candidate next steps using the LLM
    - **Evaluate** = use a value model (PRM) to score partial paths
    - **Backpropagate** = update node values up to root

    True reward comes at the end (is the final answer correct?), but expanding every path to completion is exponentially expensive. The solution is a **value model** that estimates final reward from a partial state — "how likely is this incomplete reasoning to lead to a correct answer?" This lets you prune bad paths early without finishing them.

    An **ORM (Outcome Reward Model)** only scores final answers — can't guide search mid-reasoning. A **PRM (Process Reward Model)** scores each intermediate step, which is what MCTS needs. To train a PRM: collect reasoning traces with step-by-step labels (human annotation, or automatic by checking if a step leads to correct final answers), then train the model to predict "is this step on the right track?" At search time, PRM scores each node without completing the full trajectory.

    **Example walkthrough:** Query is "What is 17 × 24?"

    ```
    Iteration 1:
      Root (query) → expand → generate children:
        A: "17 × 24 = 17 × 20 + 17 × 4"  (PRM score: 0.9)
        B: "17 × 24 = 400"                (PRM score: 0.2)
      Backprop scores to root.

    Iteration 2:
      Root → UCB picks A (high value) → expand A:
        A1: "= 340 + 68"                  (PRM score: 0.9)
        A2: "= 340 + 17"                  (PRM score: 0.3)
      Backprop.

    Iteration 3:
      Root → A → UCB picks A1 → expand:
        A1a: "= 408"                      (PRM score: 0.95)
      Backprop. Path A → A1 → A1a looks good.

    Iteration 4:
      UCB says: "B has low value but only 1 visit, explore it"
      Root → B → expand:
        B1: "Let me recalculate..."       (PRM score: 0.4)
      Backprop. Still worse than A path.

    Iteration 5:
      Root → A → A2 (underexplored) → expand:
        A2a: "= 357"                      (PRM score: 0.1)
      Backprop. Confirms A2 is bad.

    Final: Return path A → A1 → A1a = "408"
    ```

    Key points: tree grows unevenly (deep on A, shallow on B), UCB occasionally explores bad-looking paths to be sure, final answer comes from best path. In contrast, beam would keep top-k at each level and move forward together. MCTS dynamically chooses where to expand next.

2) Why does RLHF need an explicit KL penalty term if cross-entropy already minimizes KL divergence?

    They measure KL between **different distributions**:

    **Cross-entropy loss (SFT):**
    - $p$ = true label, $q$ = model prediction
    - KL(true label || model) — "match the correct answer"

    **KL term in RLHF (PPO phase):**
    - $\pi_\theta$ = new policy, $\pi_{ref}$ = reference/original model
    - KL(new model || original model) — "don't drift too far"

    The RL phase objective is:
    $$\max_\theta \mathbb{E}[R(y)] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

    **Why cross-entropy doesn't need extra KL:** It has a target (true labels) to anchor to. KL toward that target is implicit.

    **Why RL needs explicit KL:** No target distribution — just a reward signal. Without the penalty, the model can "reward hack": find degenerate outputs that score high reward but are nonsensical. The KL term keeps it close to the pretrained model that already produces coherent text.

# Coding Exercises

Be ready to implement from scratch and debug in real-time.

1) Implement a Multi-Head Attention block from scratch in PyTorch/NumPy without looking up documentation.

2) Manually derive and implement backpropagation for a simple 2-layer network.

3) Implement a KV cache for efficient autoregressive inference. Explain why it speeds up generation.

4) Implement positional encodings (sinusoidal and learned). What are the tradeoffs?

5) Implement beam search decoding.

6) Implement MCTS for LLM reasoning.

```python
import math
from typing import List, Callable

class Node:
    def __init__(self, state: str, parent=None):
        self.state = state          # partial response so far
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0            # average PRM score

    def ucb(self, c=1.4) -> float:
        if self.visits == 0:
            return float('inf')     # always explore unvisited
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

def mcts(
    query: str,
    llm_generate: Callable[[str], List[str]],  # generates candidate next steps
    prm_score: Callable[[str], float],          # scores partial solution
    num_iterations: int = 100
) -> str:
    root = Node(query)
    root.visits = 1

    for _ in range(num_iterations):
        # 1. SELECT: walk down tree using UCB
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.ucb())

        # 2. EXPAND: generate children for this node
        continuations = llm_generate(node.state)
        for cont in continuations:
            child = Node(node.state + cont, parent=node)
            node.children.append(child)

        # 3. EVALUATE: score one child with PRM
        if node.children:
            child = node.children[0]  # or random choice
            score = prm_score(child.state)
            child.visits = 1
            child.value = score

            # 4. BACKPROPAGATE: update ancestors
            current = child
            while current.parent:
                current.parent.visits += 1
                current.parent.value += score
                current = current.parent

    # Return best path: follow highest-value children from root
    node = root
    while node.children:
        node = max(node.children, key=lambda n: n.value / max(n.visits, 1))
    return node.state
```

Key components: Node with state/visits/value, UCB for selection, expand with LLM, evaluate with PRM, backpropagate scores up the tree.

# Appendix: Detailed Derivations

## MSE from Gaussian MLE

Assume targets have Gaussian noise around the true value:
$$y_i = w^T x_i + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

This means $y_i | x_i \sim \mathcal{N}(w^T x_i, \sigma^2)$. The likelihood of observing all data:
$$L(w) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^T x_i)^2}{2\sigma^2}\right)$$

Take the log:
$$\log L(w) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(y_i - w^T x_i)^2$$

To maximize log-likelihood, we minimize:
$$\sum_{i=1}^{N}(y_i - w^T x_i)^2 = \text{MSE (up to constant)}$$

## Logistic Regression Gradient

For binary cross-entropy loss $L = -[y\log p + (1-y)\log(1-p)]$ where $p = \sigma(z)$ and $z = w^T x$ (bias absorbed into $w$).

Using the chain rule:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Each term:**
1. $\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$ (from the log terms)
2. $\frac{\partial p}{\partial z} = \sigma(z)(1-\sigma(z)) = p(1-p)$ (sigmoid derivative)
3. $\frac{\partial z}{\partial w} = x$

**Multiply and simplify:**
$$\frac{\partial L}{\partial w} = \left[-\frac{y}{p} + \frac{1-y}{1-p}\right] \cdot p(1-p) \cdot x$$

$$= \left[-y(1-p) + (1-y)p\right] \cdot x$$

$$= \left[-y + yp + p - yp\right] \cdot x = (p - y) \cdot x$$

**Result (single sample):** $\frac{\partial L}{\partial w} = (p - y) \cdot x$

**For N samples:** $\frac{\partial L}{\partial w} = \sum_{i=1}^{N} (p_i - y_i) \cdot x_i$

The sigmoid derivative and log cancel nicely to give this clean form.

## Kernel Methods

Kernel methods (mainly SVMs) are an alternative to neural networks for nonlinear classification. They're mostly historical now but worth understanding.

**Context:** Logistic regression has linear decision boundaries. One way to get nonlinear boundaries is to manually add features like $x^2$, $x_1 x_2$, etc. But this gets expensive as you add more terms.

**How SVMs differ from logistic regression:**
- Logistic regression: learn weights $w$, predict via $w^T x$, throw away training data
- SVMs: store key training points ("support vectors"), predict by comparing new points to them

To classify a new point $x$:
$$\hat{y} = \text{sign}\left(\sum_i \alpha_i y_i \cdot (x_i \cdot x)\right)$$

where $x_i$ are stored support vectors, $y_i$ are their labels, $\alpha_i$ are learned weights. You're asking "is $x$ more similar to positive or negative training examples?" Similarity = dot product.

**How are support vectors chosen?** Training finds the boundary with the largest margin (distance to nearest points). The points closest to this boundary are the support vectors — they're the only ones that matter. Most points get $\alpha_i = 0$.

**Problem:** This only gives linear boundaries (dot product = linear similarity). For nonlinear, you'd transform features first: $x \to \phi(x)$.

Example: $x = [x_1, x_2] \to \phi(x) = [x_1, x_2, x_1^2, x_2^2, x_1 x_2, ...]$

Then use $\phi(x_i) \cdot \phi(x)$ as similarity. But $\phi$ can have huge or infinite dimensions — expensive to compute.

**The kernel trick:** Find a function $K$ that computes the dot product in transformed space *without* computing $\phi$:
$$K(x_i, x) = \phi(x_i) \cdot \phi(x)$$

**Example — RBF kernel:**
$$K(x_i, x_j) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$$

This simple formula is equivalent to a dot product in *infinite*-dimensional feature space. You get nonlinear boundaries without computing infinite features.

**Why neural networks won:** Kernel methods require computing $K(x_i, x_j)$ for all pairs — $O(N^2)$ cost. Neural networks scale better to large datasets and learn features end-to-end.
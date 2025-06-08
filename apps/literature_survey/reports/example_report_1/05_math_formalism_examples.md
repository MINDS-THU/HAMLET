# 5. Mathematical Formalism and Worked Example(s)

## 5.1 Bayesian Inference as a Formalism for In-Context Learning

Large language models (LLMs), such as transformers, perform *in-context learning* (ICL): they ingest a prompt consisting of input-output pairs [(x1, y1), ...,(xn, yn)] (the *context*) and predict the output for a new input x_{n+1}. A guiding hypothesis in recent literature is that LLM ICL approximates Bayesian inference on tasks seen in the context window [Aky¨¹rek et al., 2022; von Oswald et al., 2022; Xie et al., 2022].

### 5.1.1 Bayesian Posterior Predictive
Let a dataset be $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$. In the Bayesian framework, given prior $p(\theta)$ over parameters $\theta$ and likelihood $p(y|x,\theta)$, the Bayesian agent computes:

- **Posterior:**
  $$p(\theta|\mathcal{D}) \propto p(\theta) \prod_{i=1}^n p(y_i|x_i, \theta)$$
- **Posterior predictive:**
  $$p(y_{n+1}|x_{n+1}, \mathcal{D}) = \int p(y_{n+1}|x_{n+1}, \theta)p(\theta|\mathcal{D}) d\theta$$

This formalism describes how the agent updates beliefs with data $\mathcal{D}$ and predicts new outputs for test inputs¡ªsee [Section 3.2](03_bayesian_icl.md).

### 5.1.2 Amortized Bayesian Inference in Transformers

Transformers, when presented with prompt $\mathcal{C} = \{(x_1, y_1), ..., (x_n, y_n), x_{n+1}\}$ (no explicit parameter updates), produce output:

$$\hat{y}_{n+1} = f_{\mathrm{ICL}}(\mathcal{C})$$

Recent theoretical and empirical work [Aky¨¹rek et al., 2022](https://arxiv.org/abs/2210.10243); [Xie et al., 2022](https://arxiv.org/abs/2205.12667); [von Oswald et al., 2022](https://arxiv.org/abs/2202.08791) have shown that in simple tasks, $f_{\mathrm{ICL}}$ closely approximates the Bayesian posterior predictive. The transformer acts as an *amortized inference engine*, using its weights (trained on many tasks) to simulate Bayesian updating in a single forward pass.

---

## 5.2 Worked Example: Bayesian Linear Regression via In-Context Learning

Let us illustrate Bayesian ICL by walking through Bayesian linear regression, a canonical example used in recent studies [Xie et al., 2022; von Oswald et al., 2022].

### 5.2.1 Problem Setup

- Each data point: $y = x^T \theta + \epsilon$
  - $x \in \mathbb{R}^d$
  - Unknown weights: $\theta \sim \mathcal{N}(0, \tau^2 I)$ (Gaussian prior)
  - Noise: $\epsilon \sim \mathcal{N}(0, \sigma^2)$
  - $n$ observed examples $\{(x_i, y_i)\}_{i=1}^n$

### 5.2.2 Bayesian Posterior and Predictive Equations
Given prior $\mathcal{N}(0, \tau^2 I)$ and likelihood, the posterior over $\theta$ after seeing $\mathcal{D}$ is:

$$p(\theta|\mathcal{D}) = \mathcal{N}(\mu_n, \Sigma_n)$$
where:
$$
\Sigma_n = \left(\frac{1}{\tau^2}I + \frac{1}{\sigma^2} X^T X\right)^{-1}, \qquad  \mu_n = \Sigma_n \frac{1}{\sigma^2} X^T y
$$
where $X = [x_1^T; ... ; x_n^T] \in \mathbb{R}^{n \times d}$ and $y = [y_1, ..., y_n]^T$.

The **posterior predictive** for a new $x_{n+1}$:
$$
p(y_{n+1}|x_{n+1},\mathcal{D}) = \mathcal{N}(x_{n+1}^T \mu_n,\; x_{n+1}^T \Sigma_n x_{n+1} + \sigma^2)
$$

---

### 5.2.3 Numerical Illustration (Toy Example)
(Assume $d=1$, $\tau=1$, $\sigma=0.5$, $n=2$; $x_1=0$, $y_1=1$; $x_2=1$, $y_2=2$)

- $X = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
- $y = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

Calculate:
$$
\Sigma_2 = \left(1 + \frac{1}{0.25}(0^2 + 1^2) \right)^{-1} = (1 + 4)^{-1} = 0.2
$$
$$
\mu_2 = 0.2 \cdot \left(\frac{1}{0.25}(0\cdot1 + 1\cdot2)\right) = 0.2 \cdot 8 = 1.6
$$
Thus, for $x_3 = 2$:
$$
\mathbb{E}[y_3|x_3=2, \mathcal{D}] = 2 \times 1.6 = 3.2
$$
And predictive variance: $2^2 \times 0.2 + 0.25 = 0.8 + 0.25 = 1.05$

---

### 5.2.4 How Does a Transformer Perform This?

#### Mechanisms and Mapping
- **Prompt tokenization/embedding:** Model receives [(0,1), (1,2), 2] as context tokens.
  - Each token (or token-pair) represents a (feature, response), embedded in vector space.
- **Pattern recognition via attention:** Attention heads, including *induction heads* [Olsson et al., 2022], detect shared structure among examples.
  - In the regression setting, heads soft-match $x_{n+1}$ with $x_i$ in context, implicitly weighing similar past examples when predicting $y_{n+1}$.
- **Meta-learning update:** Model's parameters (meta-learned) encode a general algorithm for function approximation; in regression, this may realize the closed-form Bayesian estimator above (at least approximately).

#### Amortization in ICL
- The transformer does **not** compute $\mu_n$, $\Sigma_n$ explicitly. Instead, pattern extraction, extrapolation, and averaging occur in the forward computation, implicitly simulating these updates [Aky¨¹rek et al., 2022].
- Empirically, predictions closely match Bayesian regression, especially for simple synthetic tasks [Xie et al., 2022].

---

### 5.2.5 Diagram: Bayesian ICL Pipeline

**[Figure]: Diagram description**
- Upper row: *Bayesian flow*. Shows arrows: (Prior + Likelihood/Data) -> Posterior ($\theta|\mathcal{D}$) -> Predict $y_{n+1}$ for new $x_{n+1}$.
- Lower row: *Transformer ICL*: Sequence of context tokens ($(x_1,y_1)$,...) processed by model via embeddings and self-attention -> Internal state -> Output $\hat{y}_{n+1}$. Arrows illustrate that information from each context point is "attended" to when predicting the new point.

---

## 5.3 Summary: Mechanistic and Practical Takeaways
- **Bayesian formalism:** ICL in LLMs can closely approximate posterior predictive inference for tasks like regression, especially when prompts are aligned with training distribution.
- **Worked example:** For Bayesian linear regression, transformer ICL empirically reproduces predictive means and variances, matching theoretical posteriors in small/clean scenarios.
- **Mechanism:** Attention and meta-learned forward computation serve as implicit Bayesian updating engines, with components such as induction heads facilitating effective information aggregation from context.
- **Caveats:** Deviations from perfect Bayesian behavior occur on real-world or more complex tasks ([Falck et al., 2024](https://arxiv.org/abs/2406.00793)).

---

**References:**
- von Oswald et al., 2022. [Transformers are Bayesian Sequence-to-Sequence Learners](https://arxiv.org/abs/2202.08791)
- Xie et al., 2022. [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2205.12667)
- Olsson et al., 2022. [In-Context Learning and Induction Heads](https://arxiv.org/abs/2206.04615)
- Aky¨¹rek et al., 2022. [What Learning Algorithm is in-context Learning? Investigations with Linear Models](https://arxiv.org/abs/2210.10243)
- Falck et al., 2024. [Is In-Context Learning in LLMs Bayesian? A Martingale Perspective](https://arxiv.org/abs/2406.00793)

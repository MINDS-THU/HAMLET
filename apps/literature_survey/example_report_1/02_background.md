# 2. Background

## 2.1 What is In-Context Learning?

In-Context Learning (ICL) is an emergent capability of large language models (LLMs) wherein they exhibit the ability to learn tasks from examples presented in their input context, without the need for gradient-based parameter updates. Instead of updating parameters to store knowledge (as in traditional supervised learning), ICL enables the model to infer a function or task directly from a set of demonstrations provided within the prompt. This paradigm is highlighted by the following elements:

- **Definition**: ICL refers to the process where a model, given a prompt containing input-output exemplars for a task, predicts the output for a new query based on these contextually provided examples\(^{[1][2]}\).
- **Formalism**: Given a prompt consisting of $n$ input-output pairs $\mathcal{C} = \{(x_i, y_i)\}_{i=1}^n$ (the context), and a query input $x_{n+1}$, the model produces $\hat{y}_{n+1}$ as:
  \[
    \hat{y}_{n+1} = f_{\text{ICL}}((x_1, y_1), \ldots, (x_n, y_n), x_{n+1})
  \]
  where $f_{\text{ICL}}$ is implicitly encoded in the model weights and uses only the input context $\mathcal{C}$ for adaptation.
- **Examples**:
  - **Few-shot classification**: Providing the model with several (input, label) pairs, e.g., "cat : animal, carrot : vegetable, dog : animal, lettuce : ?", expecting the model to output "vegetable".
  - **Text generation**: Demonstrating a style or pattern in a prompt and expecting generation in the same style.
- **Contrast to Parameter Learning**: In traditional parameter learning, knowledge is incorporated into model weights via training data and optimization. In ICL, adaptation to novel tasks is accomplished solely through conditioning on the prompt, leaving parameters unchanged. This difference is critical for understanding the flexibility and generalization abilities of LLMs.

For further reading, see [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234)[1] and [In-Context Learning in Large Language Models: A Comprehensive Survey](https://www.researchgate.net/publication/382222768_In-Context_Learning_in_Large_Language_Models_A_Comprehensive_Survey)[2].

## 2.2 Overview of Large Language Models (LLMs)

Large Language Models like GPT-3 and GPT-4 are primarily built on the transformer architecture, which leverages attention mechanisms to process and generate text. Key components relevant to understanding ICL include:

- **Transformer Model Architecture**: At its core, a transformer model consists of layers of self-attention and feed-forward neural networks. Each layer enables the model to attend to different parts of the input sequence when making predictions. The ubiquitous self-attention mechanism is mathematically defined as follows:

  Given an input sequence matrix $X$, define queries ($Q$), keys ($K$), and values ($V$):
  \[
    Q = X W_q,\quad K = X W_k,\quad V = X W_v
  \]
  The self-attention output is:
  \[
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
  \]
  where $W_q, W_k, W_v$ are learned weights and $d_k$ is the dimensionality of the keys.

- **Multi-Head Attention**: This process is repeated in parallel across several "heads," allowing the model to simultaneously attend to information from different representation subspaces.

- **Prompt-Based Input & ICL**: LLMs process input as a contiguous sequence of tokens¡ªthe prompt. For ICL, the prompt includes task instructions and input-output examples. This flexible, text-based prompting mechanism is what underpins ICL: the model uses the demonstration examples given as part of its sequential context to infer and generalize patterns necessary for new predictions\(^{[3][4]}\).

- **Relevance to ICL**: The compositionality and attention mechanisms of transformers enable models to contextualize task demonstrations and queries together, thus supporting in-context adaptation without modifying model weights.

References:
- [Transformer (deep learning architecture) - Wikipedia][5]
- [Understanding and Coding the Self-Attention Mechanism - Raschka][6]
- [Prompt Engineering in Large Language Models][3]

## 2.3 Essentials of Bayesian Inference

Bayesian inference is a foundational approach in statistics and machine learning which relies on Bayes' theorem to update beliefs about unknown parameters or hypotheses in light of new evidence.

- **Probabilistic Modeling**: Let $\theta$ denote unknown model parameters and $D$ the observed data. Probabilistic modeling proceeds by specifying:
  - **Prior**: $p(\theta)$ (beliefs about $\theta$ before data)
  - **Likelihood**: $p(D \,|\, \theta)$ (data's probability given $\theta$)
  - **Posterior**: $p(\theta \,|\, D)$ (updated beliefs after observing $D$)

- **Bayes Theorem**:
  \[
    p(\theta \,|\, D) = \frac{p(D \,|\, \theta) \; p(\theta)}{p(D)}
  \]
  where $p(D) = \int p(D \,|\, \theta) p(\theta) d\theta$ is the evidence or marginal likelihood.

- **Bayesian Optimal Prediction**: Instead of selecting a single "best" parameter value, Bayesian prediction averages predictions over the posterior:
  \[
    p(y^* \,|\, x^*, D) = \int p(y^* \,|\, x^*, \theta)\, p(\theta \,|\, D) d\theta
  \]
  This reflects predictive uncertainty and often improves robustness.

- **Classical Examples in Machine Learning**:
  - **Naive Bayes Classifier**: Assumes feature independence; computes class posterior probabilities using Bayes' theorem.
  - **Bayesian Linear Regression**: Places priors over coefficients, updating them given observed data.
  - **Bayesian Neural Networks**: Models parameter uncertainty by learning posterior distributions over weights.

For further details, see [Wikipedia: Bayesian inference][7], [GeeksforGeeks: Bayes Theorem in Machine Learning][8], and [CMU Statistics PDF][9].


---

**References:**
1. [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234)
2. [In-Context Learning in Large Language Models: A Comprehensive Survey](https://www.researchgate.net/publication/382222768_In-Context_Learning_in_Large_Language_Models_A_Comprehensive_Survey)
3. [Prompt Engineering in Large Language Models](https://www.researchgate.net/publication/377214553_Prompt_Engineering_in_Large_Language_Models)
4. [arXiv:2301.00234v6](http://arxiv.org/pdf/2301.00234)
5. [Transformer (deep learning architecture) - Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))
6. [Understanding and Coding the Self-Attention Mechanism - Raschka](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
7. [Wikipedia: Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference)
8. [GeeksforGeeks: Bayes Theorem in Machine Learning](https://www.geeksforgeeks.org/bayes-theorem-in-machine-learning/)
9. [CMU Statistics PDF: Chapter 12 Bayesian Inference](https://www.stat.cmu.edu/~larry/=sml/Bayes.pdf)

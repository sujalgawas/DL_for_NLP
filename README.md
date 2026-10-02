# Deep Learning for Natural Language Processing

This repo is created to document me learning **Deep Learning for NLP** for the course:

> **Deep Learning for Natural Language Processing**
> By **Prof. Pawan Goyal**
> **IIT Kharagpur**

### Course Links

* **NPTEL Course:** https://onlinecourses.nptel.ac.in/e-learning/preview/noc26_cs181
* **Original Tutorial Links:** https://sites.google.com/view/dl4nlp-nptel/tutorials

---

## Tutorial 1

### N-gram

An N-gram is a sequence of **N words** from a given text.

The probability of a word given the previous \(N-1\) words can be approximated as:

$$
P(w_i \mid w_1, w_2, \ldots, w_{i-1})
\approx
P(w_i \mid w_{i-(N-1)}, \ldots, w_{i-1})
$$

For example, for a **bigram** model:

$$
P(w_i \mid w_{i-1})
$$

And for a **trigram** model:

$$
P(w_i \mid w_{i-2}, w_{i-1})
$$

The probability of an N-gram can be estimated using counts:

$$
P(w_i \mid w_{i-(N-1)}, \ldots, w_{i-1})
=
\frac{
C(w_{i-(N-1)}, \ldots, w_{i-1}, w_i)
}{
C(w_{i-(N-1)}, \ldots, w_{i-1})
}
$$

### Smoothing

Smoothing is used to handle **unseen N-grams**, which would otherwise have a probability of zero.

For **Laplace (Add-1) smoothing**:

$$
P_{\text{Laplace}}(w_i \mid h)
=
\frac{C(h,w_i)+1}
{C(h)+V}
$$

where:

* \(C(h,w_i)\) = count of the N-gram
* \(C(h)\) = count of the history/context
* \(V\) = vocabulary size

---

### Dataset

**Tiny Shakespeare**

---

### Perplexity

Skipped the perplexity part and just went through the comparison done in the original tutorial.

---

## Tutorial 2

### Activation Functions

Implemented and plotted various common activation functions:

* **Sigmoid**
* **Tanh**
* **ReLU**
* **Leaky ReLU**
* **Softmax**

---

### Loss Function and Gradients

Implemented a basic loss function and calculated its gradient:

* **Loss function**: \( y = x^2 \)
* **Gradient**: \( \frac{dy}{dx} = 2x \)

Also plotted the loss function to visualize it.


## Tutorial 3

This tutorial has been skipped cause it has only covered basics of pytorch

---

## Tutorial 4

### Word2Vec & Dimensionality Reduction (PCA vs. t-SNE)

* Trained a **Word2Vec** model (using Gensim on the NLTK Brown corpus) to generate distributed word representations.
* Visualized and compared word vectors in 2D using two dimensionality reduction techniques:
  * **PCA (Principal Component Analysis):** A linear dimensionality reduction technique that maximizes variance along orthogonal axes, preserving global structure.
  * **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique designed to preserve local neighborhoods and probabilistic distances, creating distinct semantic word clusters.

---

### Simple RNN with Hidden State

Implemented a character-level **Simple RNN** (`nn.RNN`) in PyTorch for next-character prediction:

* **Hidden State Initialization:** The hidden state is explicitly initialized with zeros:
  ```python
  def init_hidden(self):
      return torch.zeros(1, 1, self.hidden_size)
  ```
* **Extracting the Last Time-Step Output:** Since `nn.RNN` outputs representations for all time steps with shape `(batch_size, seq_len, hidden_size)`, slicing `[:, -1, :]` extracts the hidden representation of the final time step before feeding it into the linear classification head:
  ```python
  # out shape: (batch_size, seq_len, hidden_size)
  # out[:, -1, :] extracts the last time step representation
  out = self.fc(out[:, -1, :])
  ```

---

### Self-Attention Mechanism ($Q, K, V$)

Explored how self-attention operates using Queries ($Q$), Keys ($K$), and Values ($V$):

1. **Learnable Projections via `nn.Parameter`:**
   Created learnable weight matrices ($W_Q, W_K, W_V$) to project token embeddings into Query, Key, and Value spaces:
   ```python
   w_query = torch.nn.Parameter(torch.rand(d_q, d))
   w_key   = torch.nn.Parameter(torch.rand(d_k, d))
   w_value = torch.nn.Parameter(torch.rand(d_v, d))
   ```

2. **Attention Scores (Dot-Product with Scaling):**
   Computed the similarity score between Query and Key representations:
   $$
   \text{Scores} = \frac{Q K^T}{\sqrt{d_k}}
   $$
   In PyTorch:
   ```python
   # Raw attention scores scaled by sqrt(d_k)
   scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
   ```

3. **Attention Weights (Softmax):**
   Normalized the scores along the last dimension using softmax to obtain attention weights (probabilities summing to 1):
   $$
   \alpha = \text{Softmax}(\text{Scores})
   $$
   ```python
   attention_weights = torch.nn.functional.softmax(scores, dim=-1)
   ```

4. **Context Vector:**
   Multiplied the attention weights by the Value vectors to produce the final context representation:
   $$
   \text{Context} = \alpha V = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
   $$
   ```python
   context = torch.matmul(attention_weights, value)
   ```

> [!NOTE]
> **Scaling Factor ($\sqrt{d_k}$ vs. Simple Dot-Product):**
> * In simple dot-product attention, the score is directly computed as $Q K^T$ (`q @ k.T`).
> * In **Scaled Dot-Product Attention** (Transformers), we divide the dot product by $\sqrt{d_k}$ (or $\sqrt{\text{hidden\_size}}$). When the dimension $d_k$ is large, the dot products can grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ counteracts this effect and stabilizes gradient flow.

---

### Attention Visualization (BertViz)

* Visualized multi-head self-attention patterns interactively using **BertViz** (`model_view`, `head_view`) with `distilbert-base-uncased`.
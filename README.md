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

---

## Tutorial 5: Building a Transformer from Scratch

In this tutorial, we construct a full Transformer encoder architecture from scratch in PyTorch following the standard Hugging Face Transformer design.

---

### 1. Scaled Dot-Product Attention

* **$Q, K, V$ Matrices:** Projected from input embeddings into Query ($Q$), Key ($K$), and Value ($V$) representations.
* **Attention Score Formula:**
  $$
  \text{Scores} = \frac{Q K^T}{\sqrt{d_k}}
  $$
  where $d_k$ is the feature dimension of the key vectors (`dim_k = key.size(-1)`).
* **Batch Matrix Multiplication (`torch.bmm`):**
  Instead of simple 2D matrix multiplication, we use `torch.bmm` (batch matrix multiplication) because inputs have a 3D batch shape `(batch_size, seq_len, embed_dim)`. Hugging Face internally relies on `torch.bmm` to parallelize matrix multiplications across sequences efficiently:
  ```python
  # (batch, seq_len, dim_k) x (batch, dim_k, seq_len) -> (batch, seq_len, seq_len)
  scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
  ```
* **Softmax & Value Weighting:**
  Applying $\text{Softmax}$ over the last dimension (`dim=-1`) converts raw scores into normalized attention probabilities (weights). Multiplying these weights with the Value vectors ($V$) yields the final attended representations:
  $$
  \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
  $$
  ```python
  weights = F.softmax(scores, dim=-1)
  attn_output = torch.bmm(weights, value)
  ```
* **Implementation (with Causal Masking Support):**
  ```python
  def scaled_dot_product_attention(query, key, value, mask=None):
      dim_k = query.size(-1)
      scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
      if mask is not None:
          scores = scores.masked_fill(mask == 0, float("-inf"))
      weights = F.softmax(scores, dim=-1)
      return torch.bmm(weights, value)
  ```

---

### 2. Multi-Head Attention

* Instead of performing a single attention function across the full hidden dimension, **Multi-Head Attention** projects queries, keys, and values into multiple subspaces (`num_heads`).
* **Head Dimension:** `head_dim = embed_dim // num_heads`.
* **Concatenation & Output Projection:** We create multiple parallel `AttentionHead` instances (`nn.ModuleList`). The output vectors of all attention heads are concatenated along the feature dimension (`dim=-1`) into a single vector and projected back with an output linear layer:
  ```python
  class AttentionHead(nn.Module):
      def __init__(self, embed_dim, head_dim):
          super().__init__()
          self.q = nn.Linear(embed_dim, head_dim)
          self.k = nn.Linear(embed_dim, head_dim)
          self.v = nn.Linear(embed_dim, head_dim)

      def forward(self, hidden_state):
          return scaled_dot_product_attention(
              self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
          )

  class MultiHeadAttention(nn.Module):
      def __init__(self, config):
          super().__init__()
          embed_dim = config.hidden_size
          num_heads = config.num_attention_heads
          head_dim = embed_dim // num_heads
          self.heads = nn.ModuleList(
              [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
          )
          self.output_linear = nn.Linear(embed_dim, embed_dim)

      def forward(self, hidden_state):
          # Concatenate all attention heads into one vector along the last dimension
          x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
          x = self.output_linear(x)
          return x
  ```

---

### 3. Feed-Forward Network (FFN with GELU)

* A position-wise two-layer fully connected network applied to each token representation independently:
  $$
  \text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
  $$
* **Activation Function:** Uses **GELU** (Gaussian Error Linear Unit) instead of ReLU for smoother non-linearity (standard in BERT/Transformers), followed by dropout:
  ```python
  class FeedForward(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
          self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
          self.gelu = nn.GELU()
          self.dropout = nn.Dropout(config.hidden_dropout_prob)

      def forward(self, x):
          x = self.linear_1(x)
          x = self.gelu(x)
          x = self.linear_2(x)
          x = self.dropout(x)
          return x
  ```

---

### 4. Transformer Encoder Layer

* Combines **Multi-Head Attention** and the **Feed-Forward Network** with Pre-LayerNorm and residual (skip) connections:
  ```python
  class TransformerEncoderLayer(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
          self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
          self.attention = MultiHeadAttention(config)
          self.feed_forward = FeedForward(config)

      def forward(self, x):
          # Pre-LN with residual connection for multi-head attention
          hidden_state = self.layer_norm_1(x)
          x = x + self.attention(hidden_state)
          # Pre-LN with residual connection for feed-forward
          x = x + self.feed_forward(self.layer_norm_2(x))
          return x
  ```

---

### 5. Token & Positional Embeddings

* Self-attention is permutation-equivariant and does not naturally encode token order. Positional information must be added explicitly.
* **Dual Embeddings (`nn.Embedding`):**
  * `token_embeddings`: Maps token IDs to dense vectors (`vocab_size` $\to$ `hidden_size`).
  * `position_embeddings`: Maps position indices to dense vectors (`max_position_embeddings` $\to$ `hidden_size`).
* **Generating Position IDs:**
  * Extract the sequence length dynamically from `input_ids.size(1)`.
  * Use `torch.arange(seq_length, dtype=torch.long).unsqueeze(0)` to generate consecutive index positions `[0, 1, 2, ..., seq_len - 1]`.
* **Combining Embeddings:**
  Token and position embeddings are summed element-wise, followed by LayerNorm and Dropout:
  ```python
  class Embeddings(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
          self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
          self.layer_norm = nn.LayerNorm(config.hidden_size)
          self.dropout = nn.Dropout()

      def forward(self, input_ids):
          seq_length = input_ids.size(1)
          # Generate position IDs: shape (1, seq_len)
          position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

          token_embeddings = self.token_embeddings(input_ids)
          position_embeddings = self.position_embeddings(position_ids)

          # Element-wise addition of token + position embeddings
          embeddings = token_embeddings + position_embeddings
          embeddings = self.layer_norm(embeddings)
          embeddings = self.dropout(embeddings)
          return embeddings
  ```

---

### 6. Full Transformer Encoder

* Stacks the `Embeddings` layer and a list of `TransformerEncoderLayer` modules (`config.num_hidden_layers`):
  ```python
  class TransformersEncoder(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.embeddings = Embeddings(config)
          self.layers = nn.ModuleList(
              [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
          )

      def forward(self, x):
          x = self.embeddings(x)
          for layer in self.layers:
              x = layer(x)
          return x
  ```

---

### 7. Sequence Classification Head (`[:, 0, :]`)

* For sequence-level classification tasks (e.g. text classification, sentiment analysis), we pool the sequence representation by selecting only the hidden state of the first token — the `[CLS]` token at index 0:
  $$
  \mathbf{h}_{[\text{CLS}]} = \mathbf{H}[:, 0, :]
  $$
* Slicing `[:, 0, :]` extracts a `(batch_size, hidden_size)` tensor, which is passed through Dropout and a Linear projection layer to output class logits (`num_labels`):
  ```python
  class TransformerForSequenceClassification(nn.Module):
      def __init__(self, config):
          super().__init__()
          self.encoder = TransformersEncoder(config)
          self.dropout = nn.Dropout(config.hidden_dropout_prob)
          self.classifier = nn.Linear(config.hidden_size, config.num_labels)

      def forward(self, x):
          # Select only the hidden state of the [CLS] token at index 0
          x = self.encoder(x)[:, 0, :]
          x = self.dropout(x)
          x = self.classifier(x)
          return x
  ```

---

### 8. Causal / Autoregressive Masking

* In autoregressive / decoder architectures, future tokens are masked out to prevent a token from attending to upcoming positions.
* A lower-triangular mask (`torch.tril`) is applied, and future positions are filled with $-\infty$ using `masked_fill`, driving their post-softmax probabilities to 0:
  ```python
  seq_len = inputs.input_ids.size(-1)
  mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
  scores = scores.masked_fill(mask == 0, -float("inf"))
  ```

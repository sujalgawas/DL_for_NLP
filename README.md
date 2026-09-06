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

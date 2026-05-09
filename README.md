# Generative Adversarial Network (GAN) — Function Approximation

Training a GAN to learn and generate synthetic data points 
that approximate a custom mathematical function 
`f(x) = x² · sin(3x) / (x² + 2)` using competing 
Generator and Discriminator neural networks built with Keras.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![Task](https://img.shields.io/badge/Task-Data%20Generation-green)
![Method](https://img.shields.io/badge/Method-GAN-purple)

---

## Overview

Implements a Generative Adversarial Network (GAN) from scratch 
using Keras to learn the distribution of a custom mathematical 
function and generate synthetic data points that approximate it. 
The project demonstrates the full GAN training pipeline: a 
**Discriminator** network trained to distinguish real function 
samples from random noise, a **Generator** network trained 
to fool the Discriminator by producing increasingly realistic 
synthetic samples, and a combined GAN model where both networks 
compete in an adversarial training loop over 10,000 epochs — 
with performance evaluated every 1,000 epochs.

---

## Problem Statement

Generative Adversarial Networks are one of the most powerful 
generative modelling architectures in deep learning — used 
in image synthesis, drug discovery, data augmentation, and 
anomaly detection. This project demonstrates the core GAN 
training dynamics on a mathematically defined 1D function, 
making the concept concrete and interpretable: after training, 
the Generator produces (x, y) coordinate pairs that closely 
follow the curve `f(x) = x² · sin(3x) / (x² + 2)` — 
even though it never directly observed the function formula, 
only samples from it.

---

## Target Function

The GAN learns to approximate the following mathematical function:

```python
def func(x):
    return (x**2 * np.sin(3*x) / (x**2 + 2))
```

- **Domain:** x ∈ [-1, 1]
- **Output:** A smooth non-linear curve combining 
  polynomial and sinusoidal components
- **Real samples:** (x, f(x)) coordinate pairs 
  drawn from this function — labelled as `1` (real)
- **Fake samples:** Random (x, y) coordinate pairs 
  — labelled as `0` (fake), initially

---

## Dataset

This project uses **no external dataset**. All data 
is generated programmatically:

| Data Type | Description | Label |
|-----------|-------------|-------|
| Real samples | (x, f(x)) pairs — x drawn from Uniform[-1, 1], y computed from the target function | 1 (real) |
| Fake samples (initial) | Random (x, y) pairs — both x and y drawn from Uniform[-1, 1] | 0 (fake) |
| Latent points | Random 5-dimensional noise vectors fed into the Generator | — |
| Generated samples | (x, y) pairs produced by the trained Generator from latent noise | 0 during training, aims for 1 |

> **No download required.** All data is generated 
> within the notebook using NumPy.

---

## Approach

### Step 1 — Define the Target Function
- Implemented `func(x) = x² · sin(3x) / (x² + 2)` 
  using NumPy and Python's `math.sin`
- Plotted the target curve over x ∈ [-1, 1] to 
  visualise the distribution the GAN must learn

### Step 2 — Build the Discriminator

```python
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(100, activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
```

| Component | Details |
|-----------|---------|
| Input | 2D point (x, y) |
| Hidden layer | 100 units, ReLU, He uniform initialisation |
| Output | 1 unit, Sigmoid (real=1, fake=0) |
| Loss | Binary Cross-Entropy |
| Optimiser | Adam |

### Step 3 — Generate Real and Fake Samples

**Real samples** — points on the target function:
```python
X1 = np.random.uniform(-1, 1, n)   # x values
X2 = func(X1)                       # y = f(x) values
y  = np.ones((n, 1))                # label: real
```

**Fake samples** — random noise (pre-generator):
```python
X1 = np.random.uniform(-1, 1, n)   # random x
X2 = np.random.uniform(-1, 1, n)   # random y
y  = np.zeros((n, 1))              # label: fake
```

### Step 4 — Train the Discriminator Standalone
- Trained for 20 epochs on alternating real 
  and fake batches of 64 samples each
- Verified Discriminator learns to separate 
  real function points from random noise

### Step 5 — Build the Generator

```python
def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(50, activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model
```

| Component | Details |
|-----------|---------|
| Input | 5-dimensional latent noise vector |
| Hidden layer | 50 units, ReLU, He uniform initialisation |
| Output | 2 units, Linear (generates x, y coordinates) |
| Note | Not compiled standalone — trained through GAN |

### Step 6 — Build the Combined GAN Model

```python
def full_gan(generator, discriminator):
    discriminator.trainable = False   # freeze discriminator
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model
```

Freezing the Discriminator inside the GAN is critical — 
during Generator training, only the Generator weights 
are updated. The Discriminator provides the error signal 
but its weights remain fixed.

### Step 8 — Evaluate and Generate Synthetic Data
- Plotted Generator output (500 points) **before** 
  training — random scatter with no structure
- Evaluated Discriminator accuracy on real vs 
  generated samples every 1,000 epochs using 
  `summarize_performance()`
- Plotted Generator output (200 points) **after** 
  10,000 epochs — points should cluster along the 
  target function curve
- Saved architecture diagrams:
  - `discriminator_plot.png`
  - `gan_plot.png`

---

## Results

| Metric | Value |
|--------|-------|
| Total Training Epochs | 10,000 |
| Batch Size | 256 (128 real + 128 fake) |
| Evaluation Frequency | Every 1,000 epochs |
| Latent Dimension | 5 |

> A well-trained GAN converges to approximately 
> **50% accuracy on both real and fake samples** — 
> meaning the Discriminator can no longer 
> distinguish generated points from real ones. 
> This is the Nash Equilibrium of GAN training.


---

## Technologies Used

- **Language:** Python 3
- **Deep Learning:** TensorFlow, Keras 
  (Sequential, Dense, plot_model)
- **Data Generation:** NumPy
- **Visualisation:** Matplotlib
- **Utilities:** pydot (for model architecture plots)

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/OyelolaIbrahim/gan-function-approximation.git
cd gan-function-approximation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook gan.ipynb
```



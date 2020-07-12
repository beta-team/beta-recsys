# Models

## How to build your models under torch_engine.py

### MF

**Matrix Factorization (MF)** used in recommender systems decomposes user-item interaction matrix $D \in R^{m \times n}$ into the product of two lower dimensionality rectangular matrices $U \in R^{m \times k}$ and $V \in R^{n \times k}$, making the product of  $UV^T$ close to the real $D$ matrix as much as possible. Matrix Factorization is the mainstream algorithm based on implicit factors.

### GMF

**Generalized Matrix Factorization (GMF)** is the weighted output of dot product of the embedding vectors of user and item after activation function. Let $f$ denote active function, $e_u$ denote the embedding vector of user, $e_i$ denote the embedding vector of item, $h$ denote the weights of linear function, then the result of GMF is :
$$
z^{GMF}=f(h^T(e_u · e_i))
$$
GMF usually deals with the problem of linear interaction.

### MLP

**Multi-Layer Perceptrons (MLPs)** is a class of feedforward artificial neural network, consisting at least an input layer, a hidden layer and an output layer. Assume a MLP model has $L$ layers, $W_i (0<i < L)$ denotes the weight matrix of $i$ layer, $b_i$ denotes the $i$ bias of MLPs, $f_i$ denotes the activate function of $i$ layer, then the result of MLPs is :
$$
z^{MLP}=f_L(W_{L}^{T}(f_{L-1}(\dots f_1(W_1^TE(e_u,e_i)+b_1)\dots))+b_L)
$$
$E(e_u,e_i)$ denotes the concatenation of embedding vectors of user and item. MLPs usually deals with the problem of non-linear interaction.

### NCF

**Neural Collaborative Filtering (MCF)** is based on **GMF** and **MLP**. Let $z^{GMF}$ denote the result vector of **GMF**, $z^{MLP}$ denote the result vector of **MLP**, then the result of **NCF** is :
$$
z^{NCF}=\sigma(h^T\begin{bmatrix} z^{GMF} \\ z_{MLP} \end{bmatrix})
$$
where $h$ denotes the weights of **NCF**.

### NGCF

**Neural Graph Collaborative Filtering (NGCF)** 

### LIGHT_GCN

### CMN

### Triple2vec

### VBCAR

### NARM

### VLML

### PAIRWISE_GMF

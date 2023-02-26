# Introduction 

Nowadays recommendation systems are everywhere in various social media platforms. In
general, a recommendation system utilizes user data to recommend items that the user may
have an interest in. For example, a movie platformlike Netflix will display videos that are related
to what you have watched recently, a music app like Spotify will recommend new songs everyday
based on your previous preferences, these are contributions of recommendation systems.

Numerous methods have been developed for recommendation systems and they mainly fall
into three categories: collaborative filtering, content-based filtering, and hybrid approaches.
Collaborative filtering is a method assuming that similar users will have similar interests and
thus recommend items that have been liked by similar users, content-based filtering explores
items with similar content and recommends users similar items to what they have liked.


## Model

This project is focused on **collaborative filtering** approaches, most of which could be modeled
as a matrix completion problem where the matrix is a user-item matrix composed of each
user’s rating for each item. 

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/movie.jpeg" width="600">
  <img alt="Popular algorithms of recommendation systems." src="./images/movie.jpeg">
</picture>

*A Survey of Matrix Completion Methods for Recommendation Systems*
(A. Ramlatchan, M. Yang, Q. Liu, M. Li, J. Wang and Y. Li) gives a comprehensive summary
to various matrix completion methods used in recommendation systems, this project is going
to focus on the matrix factorization models. 


The matrix factorization models are based on the idea of **Singular Value Decomposition**, 
which decompose the sparse data matrix into a userfeature matrix and an item-feature matrix which capture 
the latent relationship between users and items, then uses the dot product of these latent vectors 
to obtain the prediction at a certain entry.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/SVD.jpeg" width="600">
  <img alt="Popular algorithms of recommendation systems." src="./images/SVD.jpeg">
</picture>



# Matrix Factorization

Now our problem of interest is, given a large and sparse data matrix $\mathbf{A} \in \mathbb{R}^{m\times n}$, how can we recover it and get a complete matrix? This section will explain the three main methods my project investigates. 

Given a data matrix $\mathbf{A} \in \mathbb{R}^{m\times n}$ where the rows of $\mathbf{A}$ correspond to m users and the columns of $\mathbf{A}$ correspond to n items, the matrix factorization methods aim to find two matrices 

$$\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_k] \in \mathbb{R}^{m\times k}, \mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k]\in \mathbb{R}^{n\times k}$$

such that $\mathbf{A} = \mathbf{U}\mathbf{V}^T$. Here $\mathbf{U}$ is regarded as a user-feature matrix composed of user-latent vectors 
$\mathbf{u}\_i \in \mathbb{R}^m, 1\leqslant i\leqslant k$ and $\mathbf{V}$ is regarded as an item-feature matrix composed of item-latent vectors $\mathbf{v}\_j \in \mathbb{R}^m, 1\leqslant j\leqslant k$, so that the $(i,j)$ entry of the data matrix $\mathbf{A}$ is regarded as the i-th user's score for the j-th item, which satisfies $\mathbf{A}\_{ij} = \mathbf{u}\_i \cdot \mathbf{v}\_j$. This means the dot product between the i-th user-latent vector $\mathbf{u}\_i$ and the j-th item-latent vector $\mathbf{v}\_j$ will give us the estimation of the score $\mathbf{A}\_{ij}$.


## Direct SVD
A straightforward idea to achieve the decomposition $\mathbf{A} = \mathbf{U}\mathbf{V}^T$ is to directly use singular value decomposition (SVD). We first fill in the missing entries of the data matrix $\mathbf{A}$ based on some prior knowledge, here we use the average score for each item (average over rows for each column) to fill in the missing entries and obtain a complete matrix $\tilde{\mathbf{A}}$; then we apply singular value decomposition $\tilde{\mathbf{A}} = \tilde{\mathbf{U}}\mathbf{\Sigma}\tilde{\mathbf{V}}^T$; since larger singular values are thought to correspond to more important latent features, we truncate the matrices $\tilde{\mathbf{U}},\tilde{\mathbf{V}}$ to be $\hat{\mathbf{U}},\hat{\mathbf{V}}$, which are composed of the first $k$ columns in $\tilde{\mathbf{U}},\tilde{\mathbf{V}}$ respectively. Now we could define the user-feature matrix $\mathbf{U} = \hat{\mathbf{U}}\mathbf{\Sigma}^{1/2} \in \mathbb{R}^{m\times k}$, and the item-feature matrix $\mathbf{V} = \hat{\mathbf{V}}\mathbf{\Sigma}^{1/2} \in \mathbb{R}^{n\times k}$, which will give us the recovered complete matrix 

$$\hat{\mathbf{A}} = \mathbf{U}\mathbf{V}^T = \hat{\mathbf{U}}\mathbf{\Sigma}^{1/2}(\hat{\mathbf{V}}\mathbf{\Sigma}^{1/2})^T = \hat{\mathbf{U}}\mathbf{\Sigma}\hat{\mathbf{V}}^T = \tilde{\mathbf{U}}_{1:k}\mathbf{\Sigma}\tilde{\mathbf{V}}_{1:k}^T$$


## Optimization Model
Although singular value decomposition is straightforward, it requires the latent vectors to be orthogonal, which is not necessary. A more general and probably more accurate approach is to construct the matrix factorization model as an optimization problem that minimizes the difference between $\mathbf{A}$ and $\mathbf{U}\mathbf{V}^T$. Let $\Omega$ denote the set of known entries in $\mathbf{A}$, then the loss function is formed as

$$E = \frac{1}{2}\underset{(i,j)\in \Omega}{\sum_{i=1}^m\sum_{j=1}^n}(A_{ij}-\mathbf{u}_i\mathbf{v}_j^T)^2 = \frac{1}{2}\underset{(i,j)\in \Omega}{\sum_{i=1}^m\sum_{j=1}^n}(A_{ij}-\sum_{r=1}^{k}\mathbf{u}_{ir}\mathbf{v}_{jr})^2 = \frac{1}{2}\underset{(i,j)\in \Omega}{\sum_{i=1}^m\sum_{j=1}^n}e_{ij}^2.$$

Minimizing $E$ could be achieved by gradient descent method where the gradients:

$$\frac{\partial E}{\partial U_{pq}} = - \underset{(p,j)\in \Omega}{\sum_{j=1}^{n}}e_{pj}v_{jq} ,\qquad \frac{\partial E}{\partial V_{pq}} = - \underset{(i,p)\in \Omega}{\sum_{j=1}^{n}}e_{ip}u_{iq}.$$

With the appropriate step size (learning rate) $\alpha$, we simultaneously update the two matrices in each iteration:

$$\mathbf{U} \leftarrow \mathbf{U} - \alpha\nabla_{u} ,\qquad \mathbf{V} \leftarrow \mathbf{V} - \alpha\nabla_{v}.$$




# Experiments

The outline of the numerical experiments in this project is to first code matrix completion solvers based on the two different models mentioned above: direct singular value decomposition and matrix factorization; then implement them on one set of artificial data several times to choose the optimal hyperparameters involved in the algorithms; finally explore different factors that might influence matrix completion error, as well as compare the accuracy and efficiency of different algorithms.

In order to measure the performances of our algorithms, every time we first randomly generate a $50\times 10$ matrix $\mathbf{M}$ as the ground-truth complete matrix, and the matrix is also of a certain fixed rank; then we set a probability $p$ of each entry in $\mathbf{M}$ to be observed and sample on $\mathbf{M}$ to obtain an artificial incomplete data matrix $\mathbf{A}$. Now we could apply different matrix completion solvers to the incomplete data matrix $\mathbf{A}$ and obtain the recovered complete matrix $\mathbf{X}$. By this mean, we are able to measure the performances of these algorithms by computing the errors between the complete matrix $\mathbf{M}$ and recovered matrix $\mathbf{X}$.  

Choosing the hyperparameters for these algorithms is subtle but very important. Both direct SVD model and matrix factorization model need us to choose an appropriate $k$ which is the length of the latent vectors where $\mathbf{U} \in \mathbb{R}^{m\times k}, \mathbf{V} \in\mathbb{R}^{n\times k}$. If $k$ is too small, the error is expected to be very large since only a few features have been captured in $\mathbf{U}$ and $\mathbf{V}$, then $\mathbf{U}\mathbf {V}^T$ can not represent the whole complete matrix; if $k$ is too large (up to $\min\{m,n\}$), $\mathbf{U},\mathbf{V}$ might contain some unnecessary information since in real life the data matrix is expected to be sparse, and it will unnecessarily slow down our algorithm. In most of the experiments, for data matrices of the size $50\times 10$, we choose $k=8$, we also choose $k$ based on the rank of the matrices in some situations. The last hyperparameter we need to care about is the step size (learning rate) $\alpha$ of the gradient descent method in the matrix factorization algorithm. For convenience, in our experiments we use constant step size instead of adaptive step size. If $\alpha$ is too small, it would take too long for the algorithm to find the local minimum; if $\alpha$ is too large, it is possible for the algorithm to bypass the local minimum every time and never find it. In most of the experiments, we use $\alpha = 0.0005$ and there are also circumstances when we need to adjust the value of $\alpha$.

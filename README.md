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

Now our problem of interest is, given a large and sparse data matrix $\mathbf{A} \in \Re^{m\times n}$, how can we recover it and get a complete matrix? This section will explain the three main methods my project investigates. 

Given a data matrix $\mathbf{A} \in \Re^{m\times n}$ where the rows of $\mathbf{A}$ correspond to m users and the columns of $\mathbf{A}$ correspond to n items, the matrix factorization methods aim to find two matrices $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_k] \in \Re^{m\times k}, \mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k]\in \Re^{n\times k}$ such that $\mathbf{A} = \mathbf{U}\mathbf{V}^T$. Here $\mathbf{U}$ is regarded as a user-feature matrix composed of user-latent vectors $\mathbf{u}_i \in \Re^m, 1\leqslant i\leqslant k$ and $\mathbf{V}$ is regarded as an item-feature matrix composed of item-latent vectors $\mathbf{v}_j \in \Re^m, 1\leqslant j\leqslant k$, so that the $(i,j)$ entry of the data matrix $\mathbf{A}$ is regarded as the i$^\text{th}$ user's score for the j$^\text{th}$ item, which satisfies $\mathbf{A}_{ij} = \mathbf{u}_i \cdot \mathbf{v}_j$. This means the dot product between the i$^\text{th}$ user-latent vector $\mathbf{u}_i$ and the j$^\text{th}$ item-latent vector $\mathbf{v}_j$ will give us the estimation of the score $\mathbf{A}_{ij}$.


## Direct SVD
A straightforward idea to achieve the decomposition $\mathbf{A} = \mathbf{U}\mathbf{V}^T$ is to directly use singular value decomposition \cite{SVD}. We first fill in the missing entries of the data matrix $\mathbf{A}$ based on some prior knowledge, here we use the average score for each item (average over rows for each column) to fill in the missing entries and obtain a complete matrix $\tilde{\mathbf{A}}$; then we apply singular value decomposition $\tilde{\mathbf{A}} = \tilde{\mathbf{U}}\mathbf{\Sigma}\tilde{\mathbf{V}}^T$; since larger singular values are thought to correspond to more important latent features, we truncate the matrices $\tilde{\mathbf{U}},\tilde{\mathbf{V}}$ to be $\hat{\mathbf{U}},\hat{\mathbf{V}}$, which are composed of the first $k$ columns in $\tilde{\mathbf{U}},\tilde{\mathbf{V}}$ respectively. Now we could define the user-feature matrix $\mathbf{U} = \hat{\mathbf{U}}\mathbf{\Sigma}^{1/2} \in \Re^{m\times k}$, and the item-feature matrix $\mathbf{V} = \hat{\mathbf{V}}\mathbf{\Sigma}^{1/2} \in \Re^{n\times k}$, which will give us the recovered complete matrix 

$$\hat{\mathbf{A}} = \mathbf{U}\mathbf{V}^T = \hat{\mathbf{U}}\mathbf{\Sigma}^{1/2}(\hat{\mathbf{V}}\mathbf{\Sigma}^{1/2})^T = \hat{\mathbf{U}}\mathbf{\Sigma}\hat{\mathbf{V}}^T = \tilde{\mathbf{U}}_{1:k}\mathbf{\Sigma}\tilde{\mathbf{V}}_{1:k}^T$$


## Optimization Model
Although singular value decomposition is straightforward, it requires the latent vectors to be orthogonal, which is not necessary. A more general and probably more accurate approach is to construct the matrix factorization model as an optimization problem that minimizes the difference between $\mathbf{A}$ and $\mathbf{U}\mathbf{V}^T$. Let $\Omega$ denote the set of known entries in $\mathbf{A}$, then the loss function is formed as

$$E = \frac{1}{2}\underset{(i,j)\in \Omega}{\sum_{i=1}^m\sum_{j=1}^n}(A_{ij}-\mathbf{u}_i\mathbf{v}_j^T)^2 = \frac{1}{2}\underset{(i,j)\in \Omega}{\sum_{i=1}^m\sum_{j=1}^n}(A_{ij}-\sum_{r=1}^{k}\mathbf{u}_{ir}\mathbf{v}_{jr})^2 = \frac{1}{2}\underset{(i,j)\in \Omega}{\sum_{i=1}^m\sum_{j=1}^n}e_{ij}^2.$$

Minimizing $E$ could be achieved by gradient descent method where the gradients:

$$\frac{\partial E}{\partial U_{pq}} = - \underset{(p,j)\in \Omega}{\sum_{j=1}^{n}}e_{pj}v_{jq} ,\qquad \frac{\partial E}{\partial V_{pq}} = - \underset{(i,p)\in \Omega}{\sum_{j=1}^{n}}e_{ip}u_{iq}.$$

With the appropriate step size (learning rate) $\alpha$, we simultaneously update the two matrices in each iteration:

$$\mathbf{U} \leftarrow \mathbf{U} - \alpha\nabla_{u} ,\qquad \mathbf{V} \leftarrow \mathbf{V} - \alpha\nabla_{v}.$$






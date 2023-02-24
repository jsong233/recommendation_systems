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


This project is focused on **collaborative filtering** approaches, most of which could be modeled
as a matrix completion problem where the matrix is a user-item matrix composed of each
user’s rating for each item. *A Survey of Matrix Completion Methods for Recommendation Systems*
(A. Ramlatchan, M. Yang, Q. Liu, M. Li, J. Wang and Y. Li) gives a comprehensive summary
to various matrix completion methods used in recommendation systems, this project is going
to focus on the matrix factorization models and rank minimization models which are most related
to what we have learned in MATH 510. The matrix factorization models are based on the
idea of **Singular Value Decomposition**, which decompose the sparse data matrix into a userfeature
matrix and an item-feature matrix which capture the latent relationship between users
and items, then uses the dot product of these latent vectors to obtain the prediction at a certain
entry. The rank minimization models are based on the assumption that our data matrix is of
low rank since users of similar interests will have similar ratings, the algorithm aims to minimize
the rank of the desired complete matrix under the restriction that it coincides with the
original matrix.


This project first codes matrix completion solvers based on the matrix factorization models
and rank minimization models mentioned above, then implements these solvers on artificial
data several times to choose the best hyperparameters, finally it verifies the assumption that
when there are more observed entries in a data matrix, which means when the original data
matrix is less sparse, thesematrix completion solvers will have better performances.

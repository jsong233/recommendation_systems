function [UserF,ItemF] = DSVD(A,k)
% recommendation system - direct SVD
% data matrix A: m users, n items 
[m,n] = size(A);
avI = mean(A); avU = mean(A,2);

% fill out missing elements 
% using average ratings for a product
for j = 1:n
    a = A(:,j);
    a(a==0) = avI(j);
    A(:,j) = a;
end

% matrix normalization
% subtract average ratings for a user
for i = 1:m
    A(i,:) = A(i,:) - avU(i);
end

% matrix factorization & latent factors
[U,S,V] = svd(A);
UserF = U(:,1:k) * sqrt(S(1:k,1:k));
ItemF = V(:,1:k) * sqrt(S(1:k,1:k));
end


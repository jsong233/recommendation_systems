function [ Mhat, history ] = admm_nnm(M, Omega, t)
% Still fixing
% Solves the following NNM problem via ADMM:
%   minimize    ||X||_* 
%   subject to  P_Omega(X) = P_Omega(M)
%               P_OmegaC(X) = P_OmegaC(Y)
%
% The solution will return in matrix Mhat
%
% history is a structure containing time, error, normError
%
% history.error is a vector containing error in each iteration
%
% history.normError is a scalar measuring the final norm error
%
% M is the complete data matrix
%
% Omega is the known indices chosen by sampling methods
%
% t is the over-relaxation parameter in the augmented lagrangian function

t_start = tic;

% Global constants and defaults
tol=1e-6;
MAX_ITER = 1000;

% data preprocessing
[m, n] = size(M);      
M_Omega = M .* Omega;

% initialization
X = sparse(m,n);
Y = sparse(m,n);
Z = sparse(m,n);

error = [];

% admm solver
for k = 1:MAX_ITER
    tmp = Y + M_Omega - 1/t * Z;
    [U,S,V] = svd(tmp); 
    S0 = zeros(size(S));
    l = min(size(S));
    S0(1:l,1:l) = diag(max(0,diag(S) - 1/t));
    X = U * S0 * V'; 
    tmp = X + 1/t * Z;
    tmp(Omega) = 0;
    Y = tmp;
    Z = Z + t * (X - M_Omega - Y);
    
    error = [error, norm(X - Y - M_Omega,'fro') / ...
            norm(M_Omega,'fro')];
   
    % terminate rule
    if norm(X - Y - M_Omega,'fro') / norm(M_Omega,'fro') < tol
       break;
    end 
end

history.error = error;
if length(history.error) == MAX_ITER
    sprintf('The algorithm does not converge.');
end

Mhat = X;
history.normError = norm(Mhat - M,'fro') / norm(M,'fro');
history.time = toc(t_start);
end



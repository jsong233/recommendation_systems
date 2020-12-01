function [U,V,history] = BSVD_GD(A,k,alpha)
% recommendation system - basic SVD
% using (batch) gradient descent GD
% using simple initialization
% A: rating matrix, m users, n items
% alpha: learning rate
% k: truncated size 
% A = UV' approximation
t_start = tic;
[m,n] = size(A);
S = (A ~= 0); % the observed entries in A
avA = mean(mean(A)); init = sqrt(avA./k);

U = init * ones(m,k) + rand(m,k) - 0.5; 
V = init * ones(n,k) + rand(n,k) - 0.5;
UU = zeros(m,k); VV = zeros(n,k);
E = S .* (A - U * V'); 

% matrix factorization
MAX_ITER = 1000; 
tol = 1e-5;
for t = 1:MAX_ITER
    % gradient descent step for U V
    for i = 1:m
        for j = 1:k
            UU(i,j) = U(i,j) + alpha * E(i,:) * V(:,j);
        end
    end
    for i = 1:n
        for j = 1:k
            VV(i,j) = V(i,j) + alpha * E(:,i)' * U(:,j);
        end
    end        
    % update U V
    U = UU; V = VV;

    % compute the error matrix
    for i = 1:m
        for j = 1:n
            if (S(i,j)~=0)
                E(i,j) = A(i,j) - U(i,:) * V(j,:)';
            end
        end
    end
    % check convergence condition
    error = norm(E,'fro')/norm(A,'fro');
    if (error < tol)
        break;
    end
end
if (t == MAX_ITER)
    sprintf('The algorithm does not converge.')
else
    sprintf('number of iterations = %d',t)
end
history.error = error;
history.time = toc(t_start);
end


function [U,V,history] = BSVD_SGD(A,k,alpha)
% recommendation system - basic SVD
% using stochastic gradient descent SGD
% A: rating matrix, m users, n items
% alpha: learning rate
% k: truncated size 
% A = UV' approximation
t_start = tic;
[m,n] = size(A);
S = (A ~= 0); % the observed entries in A
avI = mean(A); avU = mean(A,2);
E = zeros(m,n);

%matrix normalization
%for i = 1:m
    %A(i,:) = A(i,:) - avU(i);
%end

% initialize U V using DSVD
for j = 1:n
    a = A(:,j);
    a(a==0) = avI(j);
    A(:,j) = a;
end
[U,Sigma,V] = svd(A);
U = U(:,1:k) * sqrt(Sigma(1:k,1:k));
V = V(:,1:k) * sqrt(Sigma(1:k,1:k));
UU = zeros(m,k); VV = zeros(n,k);


% matrix factorization
MAX_ITER = 1000; 
for t = 1:MAX_ITER
    % randomly shuffle entries in S
    idS = reshape(randperm(m*n),m,n);
    SS = S(idS);
    
    for i = 1:m
        for j = 1:n
            if (SS(i,j)~=0)
                % compute error
                e = A(i,j) - U(i,:) * V(j,:)';
                
                % SGD step for U V
                UU(i,:) = U(i,:) + alpha * e * V(j,:);
                VV(j,:) = V(j,:) + alpha * e * U(i,:);
                U(i,:) = UU(i,:); V(j,:) = VV(j,:);
            end
        end
    end
end
if (t == MAX_ITER)
    sprintf('The algorithm does not converge.')
else
    sprintf('number of iterations = %d',t)
end
history.time = toc(t_start);
end


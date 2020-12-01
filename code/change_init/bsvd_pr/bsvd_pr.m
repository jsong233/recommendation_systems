% SVD recommender for different probabilities
% Fix the size of matrices to be 50 * 10
% Fix the truncated size k = 6;
% Vary the rank of matrices to be 3, 5, 7, 9
% Vary the probabilities of each observed entry to be 0.05:0.05:1


% Global constants
alpha = 0.00001;
m = 50;
n = 10;
sampleIter = 10;
rankIter = 10;

% Generate pr
pr = 0.05 : 0.05 : 1;
N = length(pr);

% Generate rank
rank = [3,5,7,9];

% Initialization
err1 = zeros(sampleIter,N,rankIter,4);
err2 = zeros(sampleIter,N,4);
Err = zeros(N,4);


for k = 1:4
    r = rank(k);

    for j = 1:rankIter
        % Generate 50*10 matrix with rank r
        M = randi([1,5],m,r) * randi([1,5],r,n);

        for i = 1:N
            p = pr(i);

            for h = 1:sampleIter
                % Generate Omega
                Omega = (rand(m,n) <= p);
                A = M .* Omega;

                % ADMM for nnm
                [U,V,~] = BSVD_GD(A,6,alpha);
                err1(h,i,j,k) = norm(M - U * V')/norm(M);
            end
        end
    end

    % average over rankIter j
    for h = 1:sampleIter
        for i = 1:N
            err2(h,i,k) = mean(err1(h,i,:,k));
        end
    end

    % average over sampleIter h
    for i = 1:N
        Err(i,k) = mean(err2(:,i,k));
    end
end


% set figure parameters
set(0,'DefaultLineLineWidth',2);

blue = [0.0000    0.4470    0.7410];
red = [0.8500    0.3250    0.0980];
gold = [0.9290    0.6940    0.1250];
teal = [32 178 170]/255;
green= [134, 179, 0]/255;
purple = [153 102 255]/255;

color = {blue red gold green teal purple};
lineSpec = {'-o','-^','-s','-*','-+','-d'};


% Plotting
figure;

% norm error
for k = 1:4
    eb = shadedErrorBar(pr,err2(:,:,k),{@mean,@std},'lineprops',...
        {lineSpec{k},'markersize',8});
    eb.patch.FaceColor = color{k};
    eb.mainLineColor = color{k};
    set(eb.edge(1), 'Color', color{k}+(1-color{k})*0.5);
    set(eb.edge(2), 'Color', color{k}+(1-color{k})*0.5);
    hold on;
end
for k = 1:4
    h(k) = plot(pr,Err(:,k),lineSpec{k},'markersize',8,'Color',color{k});
    hold on;
end
set(gca,'FontSize',24);
l = legend(h,'$r(M) = 3$','$r(M) = 5$','$r(M) = 7$','$r(M) = 9$');
set(l,'Interpreter','latex')
set(l,'FontSize',28);
set(l,'FontName','Times New Roman');
xlim([0,1]);
ylim([0,1]);
xlabel('$p$','Interpreter','latex','FontSize',36)
ylabel('$E$','Interpreter','latex','FontSize',36)
grid on;


saveas(gcf,'bsvd_pr.fig','fig');
saveas(gcf,'bsvd_pr.png','png');
save('bsvd_pr.mat')

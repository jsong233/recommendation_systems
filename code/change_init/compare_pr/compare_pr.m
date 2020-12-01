% Compare different recommendation systems 
% Fix the size of matrices to be 50 * 10
% Fix the rank of matrices to be 5
% Fix the truncated size to be k = 8
% Vary the probabilities of each observed entry to be 0.05:0.05:1

% Global constants
alpha = 0.00001;
t = 0.1;
m = 50;
n = 10;
r = 5;
sampleIter = 10;
rankIter = 10;

% Generate pr
pr = 0.05 : 0.05 : 1;
N = length(pr);


% Initialization
err1_dsvd = zeros(sampleIter,N,rankIter);
err1_bsvd = zeros(sampleIter,N,rankIter);
err1_nnm = zeros(sampleIter,N,rankIter);

err2_dsvd = zeros(sampleIter,N);
err2_bsvd = zeros(sampleIter,N);
err2_nnm = zeros(sampleIter,N);

Err_dsvd = zeros(N,1);
Err_bsvd = zeros(N,1);
Err_nnm = zeros(N,1);


for j = 1:rankIter
    % Generate 100*30 matrix with rank r
    M = randi([1,5],m,r) * randi([1,5],r,n);

    for i = 1:N
        p = pr(i);

        for h = 1:sampleIter
            % Generate Omega
            Omega = (rand(m,n) <= p);
            A = M .* Omega;

            % recommenders
            avU = mean(A,2); AVU = [];
            for k = 1:n
                AVU = [AVU,avU];
            end
            [UserF,ItemF] = DSVD(A,8); EstA = UserF * ItemF' + AVU;
            err1_dsvd(h,i,j) = norm(M - EstA,'fro')/norm(M,'fro');
            
            [U,V,~] = BSVD_GD(A,8,alpha);
            err1_bsvd(h,i,j) = norm(M - U * V','fro')/norm(M,'fro');
            
            [Mhat, history] = admm_nnm(M,Omega,t);
            err1_nnm(h,i,j) = history.normError;
        end
    end
end

% average over rankIter j
for h = 1:sampleIter
    for i = 1:N
        err2_dsvd(h,i) = mean(err1_dsvd(h,i,:));
        err2_bsvd(h,i) = mean(err1_bsvd(h,i,:));
        err2_nnm(h,i) = mean(err1_nnm(h,i,:));
    end
end

% average over sampleIter h
for i = 1:N
    Err_dsvd(i) = mean(err2_dsvd(:,i));
    Err_bsvd(i) = mean(err2_bsvd(:,i));
    Err_nnm(i) = mean(err2_nnm(:,i));
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
eb = shadedErrorBar(pr,err2_dsvd,{@mean,@std},'lineprops',...
    {lineSpec{1},'markersize',8});
eb.patch.FaceColor = color{1};
eb.mainLineColor = color{1};
set(eb.edge(1), 'Color', color{1}+(1-color{1})*0.5);
set(eb.edge(2), 'Color', color{1}+(1-color{1})*0.5);
hold on;

eb = shadedErrorBar(pr,err2_bsvd,{@mean,@std},'lineprops',...
    {lineSpec{2},'markersize',8});
eb.patch.FaceColor = color{2};
eb.mainLineColor = color{2};
set(eb.edge(1), 'Color', color{2}+(1-color{2})*0.5);
set(eb.edge(2), 'Color', color{2}+(1-color{2})*0.5);
hold on;

eb = shadedErrorBar(pr,err2_nnm,{@mean,@std},'lineprops',...
    {lineSpec{3},'markersize',8});
eb.patch.FaceColor = color{3};
eb.mainLineColor = color{3};
set(eb.edge(1), 'Color', color{3}+(1-color{3})*0.5);
set(eb.edge(2), 'Color', color{3}+(1-color{3})*0.5);
hold on;


h(1) = plot(pr,Err_dsvd,lineSpec{1},'markersize',8,'Color',color{1});
hold on;
h(2) = plot(pr,Err_bsvd,lineSpec{2},'markersize',8,'Color',color{2});
hold on;
h(3) = plot(pr,Err_nnm,lineSpec{3},'markersize',8,'Color',color{3});
hold on;


set(gca,'FontSize',24);
l = legend(h,'dsvd','bsvd-gd','nnm');
%set(l,'Interpreter','latex')
set(l,'FontSize',28);
set(l,'FontName','Times New Roman');
xlim([0,1]);
ylim([0,1]);
xlabel('$p$','Interpreter','latex','FontSize',36)
ylabel('$E$','Interpreter','latex','FontSize',36)
grid on;


saveas(gcf,'compare_pr.fig','fig');
saveas(gcf,'compare_pr.png','png');
save('compare_pr.mat')

% Compare different recommendation systems 
% Fix the size of matrices to be 100 * 30
% Fix the rank of matrices to be 5
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
err1_gd = zeros(sampleIter,N,rankIter);
err1_sgd = zeros(sampleIter,N,rankIter);
t1_gd = zeros(sampleIter,N,rankIter);
t1_sgd = zeros(sampleIter,N,rankIter);

err2_gd = zeros(sampleIter,N);
err2_sgd = zeros(sampleIter,N);
t2_gd = zeros(sampleIter,N);
t2_sgd = zeros(sampleIter,N);

Err_gd = zeros(N,1);
Err_sgd = zeros(N,1);
T_gd = zeros(N,1);
T_sgd = zeros(N,1);


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
            [U1,V1,history1] = BSVD_GD(A,8,alpha);
            [U2,V2,history2] = BSVD_SGD(A,8,alpha);
            err1_gd(h,i,j) = norm(M - U1 * V1','fro')/norm(M,'fro');
            err1_sgd(h,i,j) = norm(M - U2 * V2','fro')/norm(M,'fro');
            t1_gd(h,i,j) = history1.time;
            t1_sgd(h,i,j) = history2.time;
        end
    end
end

% average over rankIter j
for h = 1:sampleIter
    for i = 1:N
        err2_gd(h,i) = mean(err1_gd(h,i,:));
        err2_sgd(h,i) = mean(err1_sgd(h,i,:));
        t2_gd(h,i) = mean(t1_gd(h,i,:));
        t2_sgd(h,i) = mean(t1_sgd(h,i,:));
    end
end

% average over sampleIter h
for i = 1:N
    Err_gd(i) = mean(err2_gd(:,i));
    Err_sgd(i) = mean(err2_sgd(:,i));
    T_gd(i) = mean(t2_gd(:,i));
    T_sgd(i) = mean(t2_sgd(:,i));
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

subplot(1,2,1);
% norm error
eb = shadedErrorBar(pr,err2_gd,{@mean,@std},'lineprops',...
    {lineSpec{1},'markersize',8});
eb.patch.FaceColor = color{1};
eb.mainLineColor = color{1};
set(eb.edge(1), 'Color', color{1}+(1-color{1})*0.5);
set(eb.edge(2), 'Color', color{1}+(1-color{1})*0.5);
hold on;

eb = shadedErrorBar(pr,err2_sgd,{@mean,@std},'lineprops',...
    {lineSpec{2},'markersize',8});
eb.patch.FaceColor = color{2};
eb.mainLineColor = color{2};
set(eb.edge(1), 'Color', color{2}+(1-color{2})*0.5);
set(eb.edge(2), 'Color', color{2}+(1-color{2})*0.5);
hold on;


h(1) = plot(pr,Err_gd,lineSpec{1},'markersize',8,'Color',color{1});
hold on;
h(2) = plot(pr,Err_sgd,lineSpec{2},'markersize',8,'Color',color{2});
hold on;


set(gca,'FontSize',24);
l = legend(h,'bsvd-gd','bsvd-sgd');
%set(l,'Interpreter','latex')
set(l,'FontSize',28);
set(l,'FontName','Times New Roman');
xlim([0,1]);
ylim([0,1]);
xlabel('$p$','Interpreter','latex','FontSize',36)
ylabel('$E$','Interpreter','latex','FontSize',36)
grid on;


subplot(1,2,2);
eb = shadedErrorBar(pr,t2_gd,{@mean,@std},'lineprops',...
    {lineSpec{1},'markersize',8});
eb.patch.FaceColor = color{1};
eb.mainLineColor = color{1};
set(eb.edge(1), 'Color', color{1}+(1-color{1})*0.5);
set(eb.edge(2), 'Color', color{1}+(1-color{1})*0.5);
hold on;

eb = shadedErrorBar(pr,t2_sgd,{@mean,@std},'lineprops',...
    {lineSpec{2},'markersize',8});
eb.patch.FaceColor = color{2};
eb.mainLineColor = color{2};
set(eb.edge(1), 'Color', color{2}+(1-color{2})*0.5);
set(eb.edge(2), 'Color', color{2}+(1-color{2})*0.5);
hold on;


h(1) = plot(pr,T_gd,lineSpec{1},'markersize',8,'Color',color{1});
hold on;
h(2) = plot(pr,T_sgd,lineSpec{2},'markersize',8,'Color',color{2});
hold on;


set(gca,'FontSize',24);
l = legend(h,'bsvd-gd','bsvd-sgd');
%set(l,'Interpreter','latex')
set(l,'FontSize',28);
set(l,'FontName','Times New Roman');
xlabel('$p$','Interpreter','latex','FontSize',36)
ylabel('$T$','Interpreter','latex','FontSize',36)
grid on;


saveas(gcf,'sgd_pr.fig','fig');
saveas(gcf,'sgd_pr.png','png');
save('sgd_pr.mat')

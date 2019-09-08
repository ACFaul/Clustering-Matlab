% Number of samples to be generated.
N = 1000;
%Number of clusters
K = 3;
% Concentration/scaling parameter.
alpha = 10*K;
% Parametres of the normal-inverse-Wishart distribution
mu0 = [0 0];
lambda = 1/5;
Psi = [1 1.5; 1.5 3];
nu = 6;
[V,idx,MU,SIGMA] = BayesianMixture(N,K,alpha,mu0,lambda,Psi,nu);

figure;
data = gscatter(V(:,1),V(:,2),idx);
for i = 1:numel(data)
    data(i).DisplayName = strcat('Cluster', data(i).DisplayName,...
        ', size = ', string(numel(data(i).XData)));
end
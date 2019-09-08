function [V,idx,MU,SIGMA] = BayesianMixture(N,K,alpha,mu0,...
    lambda,Psi,nu)
% Input:
% N:      number of data points to be generated,
% K:      number of clusters
% alpha:  concentration parameter,
% mu0:    location vector,
% lambda: mean fraction,
% Psi:    inverse scale matrix,
% nu:     degrees of freedom.
% Output:
% V:      data vector,
% idx:    index vector which process generated the sample,
% MU:     array of the means of all processes,
% SIGMA: array of the covariance matrices of all processes.

% Generate vector of mixture coefficients from Dirichlet 
% distribution given by alpha and K using the Gamma dsitribution.
a = alpha/K*ones(1,K);
PI = gamrnd(a,1);
PI = PI/sum(PI);

% Generate latent indicator variable for cluster mambership.
idx = randsample(K,N,true,PI);

% Generate distribution for each cluster.
SIGMA = zeros(2,2,K);
MU = zeros(2,K);
for k=1:K
    sigma = iwishrnd(Psi,nu);
    SIGMA(:,:,k) = sigma;
    MU(:,k) = mvnrnd(mu0,sigma/lambda);
end

% Generate data.
V = zeros(N,2);
for n=1:N
    v = mvnrnd(MU(:,idx(n)),SIGMA(:,:,idx(n)));
	V(n,:) = v;
end

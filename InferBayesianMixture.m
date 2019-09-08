function [idx,MU,SIGMA] = InferBayesianMixture(V, K,alpha,m,...
    lambda,Psi,nu,iter) 
% Input:
% V:      data,
% K:      number of clusters
% alpha:  prior concentration parameter,
% m:      prior location vector,
% lambda: prior mean fraction,
% Psi:    prior inverse scale matrix,
% nu:     prior degrees of freedom,
% iter:   number of iterations
% Output:
% idx:    index vector of cluster assignments,
% MU:     array of the means sampled from the posterior normal 
%         inverse Wishart distributions for each cluster,
% SIGMA: array of the covariance matrices sampled from the posterior
%         normal inverse Wishart distributions for each cluster.

N = size(V,1);
% Start with random cluster assignments.
idx = randi(K,1,N);

% For each cluster store its size, posterior location vector,
% posterior mean fraction, posterior inverse scale matrix,
% posterior degrees of freedom.
nk = zeros(1,K);
M = zeros(2,K);
LAMBDA = zeros(1,K);
PSI = zeros(2,2,K);
NU = zeros(1,K);
% For each cluster store a draw from the posterior normal,
% inverse Wishart distribution.
SIGMA = zeros(2,2,K);
MU = zeros(2,K);
% For each cluster initialize these for the initial random cluster
% assignments and draw MU and SIGMA from that posterior normal,
% inverse Wishart distribution.
for k=1:K
    v = V(idx == k,:);
    nk(k) = size(v,1);
    sampleM = mean(v);
    sampleS = (nk(k)-1) * cov(v);
    M(:,k) = (lambda*m(:) + nk(k)*sampleM(:))/(lambda +nk(k));
    LAMBDA(k) = lambda + nk(k);
    PSIk = Psi(:,:) + sampleS + lambda*nk(k)*(sampleM(:) - m(:))*...
        (sampleM(:) - m(:))'/(lambda +nk(k));
    % Store Cholesky factorization to maintain positive definiteness.
    PSI(:,:,k) = chol(PSIk);
    NU(k) = nu + nk(k);
    SIGMA(:,:,k) = iwishrnd(PSI(:,:,k)'* PSI(:,:,k),NU(k));
    MU(:,k) = mvnrnd(M(:,k),SIGMA(:,:,k)/LAMBDA(k));
end
for i=1:iter
    % Consider the data in a random order.
    for n = randperm(N)
        v = V(n,:);
        l = idx(n);
        % Remove this sample from the data set and update the cluster
        % it was assigned to.
        priorlambda = LAMBDA(l)-1;
        % Rank one update on the Cholesky factorization to preserve
        % postive definiteness.
        update = sqrt(LAMBDA(l)/priorlambda) * (v' - M(:,l));
        PSI(:,:,l) = cholupdate(PSI(:,:,l),update,'-');
        M(:,l) = (LAMBDA(l)*M(:,l) - v(:))/priorlambda;
        NU(l) = NU(l)-1;
        LAMBDA(l) = priorlambda;
        SIGMA(:,:,l) = iwishrnd(PSI(:,:,l)'* PSI(:,:,l),NU(l));
        MU(:,l) = mvnrnd(M(:,l),SIGMA(:,:,l)/LAMBDA(l));
        nk(l) = nk(l)-1;
        % Calculate cluster assignment probabilities.
        p = zeros(1,K);
        for k=1:K
            p(k) = (nk(k) + alpha/K)/(N-1+alpha)*...
                mvnpdf(v',MU(:,k),SIGMA(:,:,k));
        end
        p = p/sum(p);
        % Sample new indicator variable.
        l  = randsample(K,1,true,p);
        idx(n) = l;
        % Update the cluster the sample is now assigned to.
        postlambda = LAMBDA(l)+1;
        % Rank one update on the Cholesky factorization to preserve
        % postive definiteness.
        update = sqrt(LAMBDA(l)/postlambda) * (v' - M(:,l));
        PSI(:,:,l) = cholupdate(PSI(:,:,l),update,'+');
        M(:,l) = (LAMBDA(l)*M(:,l) + v(:))/postlambda;
        NU(l) = NU(l)+1;
        LAMBDA(l) = postlambda;
        SIGMA(:,:,l) = iwishrnd(PSI(:,:,l)'* PSI(:,:,l),NU(l));
        MU(:,l) = mvnrnd(M(:,l),SIGMA(:,:,l)/LAMBDA(l));
        nk(l) = nk(l)+1;
    end
end
function [idx,K,MU,SIGMA] = InferDP(V,alpha,m,lambda,Psi,nu,iter) 
% Input:
% V:      data,
% alpha:  prior concentration parameter,
% m:      prior location vector,
% lambda: prior mean fraction,
% Psi:    prior inverse scale matrix,
% nu:     prior degrees of freedom,
% iter:   number of iterations
% Output:
% idx:    index vector of cluster assignments,
% K:      number of clusters
% MU:     array of the means sampled from the posterior normal 
%         inverse Wishart distributions for each cluster,
% SIGMA: array of the covariance matrices sampled from the posterior
%         normal inverse Wishart distributions for each cluster.

N = size(V,1);   % Number of Samples.
K = 0;           % Number of clusters.
idx =zeros(N,1); % Vector cluster assignments.
nk = [];         % Vector of cluster sizes.
% For each cluster:
M = [];        % Posterior location vector.
LAMBDA = [];   % Posterior mean fraction.
PSI = [];      % Cholesky factorization of posterior inverse 
               % scale matrix.
NU = [];       % Posterior degrees of freedom.
SIGMA = [];    % Covariance matrix draw.
MU = [];       % Mean draw.

% Use Cholesky factorization of Psi to ensure positive definiteness.
Psi = chol(Psi);

% Consider samples in a random order.
order = randperm(N);
% Initialize cluster assignments.
for n = 1:N
    v = V(order(n),:);
    % Calculate cluster assignment probabilities.
    p = zeros(1,K+1);
    for k=1:K
        p(k) = nk(k)/(n-1+alpha)*mvnpdf(v',MU(:,k),SIGMA(:,:,k));
    end
    p(K+1) = alpha/(n-1+alpha)*mvnpdf(v',m',Psi);
    p = p/sum(p);
    % Sample new indicator variable.
    l  = randsample(K+1,1,true,p);
    idx(order(n)) = l;
    if l <=K
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
    else
        % Create new cluster.
        LAMBDA = cat(2,LAMBDA,lambda+1);
        update = sqrt(lambda/(lambda+1)) * (v - m)';
        PSI = cat(3,PSI,cholupdate(Psi,update,'+'));
        M = cat(2,M,(lambda* m' +v' )/(lambda+1));
        NU = cat(2,NU,nu+1);
        SIGMA = cat(3,SIGMA,iwishrnd(PSI(:,:,l)'* PSI(:,:,l),NU(l)));
        MU = cat(2,MU,mvnrnd(M(:,l),SIGMA(:,:,l)/LAMBDA(l))');
        nk = cat(2,nk,1);
        K = K+1;
    end
end

for i = 1:iter
    % Consider the data in a random order.
    for n = randperm(N)
        v = V(n,:);
        l = idx(n);
        % Remove this sample from the data set and update the cluster
        % it was assigned to.
        nk(l) = nk(l)-1;
        if nk(l) == 0
            % Remove empty cluster.
            nk(l) = [];
            PSI(:,:,l) = [];
            M(:,l) = [];
            NU(l) = [];
            LAMBDA(l) = [];
            SIGMA(:,:,l) = [];
            MU(:,l) = [];
            % Adjust cluster numbering.
            temp = idx>l;
            idx(temp) = idx(temp)-1;
            K = K-1;
        else
            priorlambda = LAMBDA(l)-1;
            % Rank one update on the Cholesky factorization to 
            % preserve postive definiteness.
            update = sqrt(LAMBDA(l)/priorlambda) * (v' - M(:,l));
            PSI(:,:,l) = cholupdate(PSI(:,:,l),update,'-');
            M(:,l) = (LAMBDA(l)*M(:,l) - v(:))/priorlambda;
            NU(l) = NU(l)-1;
            LAMBDA(l) = priorlambda;
            SIGMA(:,:,l) = iwishrnd(PSI(:,:,l)'* PSI(:,:,l),NU(l));
            MU(:,l) = mvnrnd(M(:,l),SIGMA(:,:,l)/LAMBDA(l));
        end
        % Calculate cluster assignment probabilities.
        p = zeros(1,K+1);
        for k=1:K
            p(k) = nk(k)/(N-1+alpha)*mvnpdf(v',MU(:,k),SIGMA(:,:,k));
        end
        p(K+1) = alpha/(N-1+alpha)*mvnpdf(v',m',Psi);
        p = p/sum(p);
        % Sample new indicator variable.
        l  = randsample(K+1,1,true,p);
        idx(n) = l;
        if l <=K
            % Update the cluster the sample is now assigned to.
            postlambda = LAMBDA(l)+1;
            % Rank one update on the Cholesky factorization to 
            % preserve postive definiteness.
            update = sqrt(LAMBDA(l)/postlambda) * (v' - M(:,l));
            PSI(:,:,l) = cholupdate(PSI(:,:,l),update,'+');
            M(:,l) = (LAMBDA(l)*M(:,l) + v(:))/postlambda;
            NU(l) = NU(l)+1;
            LAMBDA(l) = postlambda;
            SIGMA(:,:,l) = iwishrnd(PSI(:,:,l)'* PSI(:,:,l),NU(l));
            MU(:,l) = mvnrnd(M(:,l),SIGMA(:,:,l)/LAMBDA(l));
            nk(l) = nk(l)+1;
        else
            % Create new cluster.
            LAMBDA = cat(2,LAMBDA,lambda+1);
            update = sqrt(lambda/(lambda+1)) * (v - m)';
            PSI = cat(3,PSI,cholupdate(Psi,update,'+'));
            M = cat(2,M,(lambda* m' +v' )/(lambda+1));
            NU = cat(2,NU,nu+1);
            SIGMA = cat(3,SIGMA,iwishrnd(PSI(:,:,l)'* ...
                PSI(:,:,l),NU(l)));
            MU = cat(2,MU,mvnrnd(M(:,l),SIGMA(:,:,l)/LAMBDA(l))');
            nk = cat(2,nk,1);
            K = K+1;
        end      
    end
end
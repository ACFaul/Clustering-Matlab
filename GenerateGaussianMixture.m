% Number of samples to be generated.
N = 100;
% Array of means.
MU = [1 1;-4 -1; 1 -2];
% Concatenation of co-variance matrices.
SIGMA = cat(3,[2 0; 0 .5],[1 0.5; 0.5 1],[1 0; 0 1]);
% Mixing coefficients.
p = [0.4 0.5 0.1];
% Gaussian mixture model.
GMModel = gmdistribution(MU,SIGMA,p);
% Generate and display data.
rng(1); % for reproducibility
[V,idx] = random(GMModel,N);
gscatter(V(:,1),V(:,2),idx,'bgr','...',[10 10 10]);

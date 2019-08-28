fit = load('test_fit.mat','sigma','gamma','beta0','beta','sigmabar');
test_params = [.8,2,1.9,1,1.5].';
out = svsa(test_params,fit);
plot(out)
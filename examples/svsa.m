function line = svsa(phi,fit)
    delta = sum(phi.^2);
    expon = fit.sigma*phi - single(fit.gamma)*ones(size(fit.sigma,1),1)*delta;
    expnn = exp(expon-fit.sigmabar);
    line = fit.beta0 + fit.beta * expnn;
end
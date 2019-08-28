get_fit = function(fname){
    fit = list()
    fit$sigma = as.matrix(read.csv(paste0(fname,'_sigma.csv'),header=FALSE))
    fit$sigmabar = unlist(read.csv(paste0(fname,'_sigmabar.csv'),header=FALSE))
    fit$beta0 = as.matrix(read.csv(paste0(fname,'_beta0.csv'),header=FALSE))[,1,drop=TRUE]
    fit$beta = as.matrix(read.csv(paste0(fname,'_beta.csv'),header=FALSE))
    fit$gamma = as.numeric(read.csv(paste0(fname,'_gamma.csv'),header=FALSE))
    return(fit)
}

svsa=function(phi,fit){
    delta = colSums(phi^2)
    sigphi = fit$sigma%*%phi
    expon = sweep(sigphi,2,fit$gamma*delta)
    expon = expon-fit$sigmabar
    expnn = exp(expon);
    line = fit$beta %*% expnn
    line = line + fit$beta0
    return(line)
}

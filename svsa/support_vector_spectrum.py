from sklearn.svm import SVR
from sklearn.svm import NuSVR
import numpy as np
import scipy.io as sio

class SupportVectorSpectrum:
    def __init__(self,train_params,train_lineshapes,train_x):
        self.params=train_params
        self.lineshapes = train_lineshapes
        self.x = train_x
    
    def fit(self,eps=1E-15,nu=None,gamma=1, K=5):

        self.gamma = gamma
        self.nu = nu
        self.K = K
        
        # SVD (FPCA) of self.lineshapes
        self.u,self.s,self.vt = np.linalg.svd(self.lineshapes,full_matrices=False)
        sv = self.s[0:K]*self.vt[0:K,:].transpose()

        # Fit the SVRs
        def fitu(params_df, u_df,mdl, *args,**kwargs):
            mod = mdl(*args,**kwargs)    
            modf = mod.fit(params_df,u_df)
            return(modf)

        if(nu is not None):
            self.mods = [fitu(self.params,self.u[:,k],mdl=NuSVR,
                         nu=self.nu,gamma=gamma)
                         for k in range(0,K)]
        else:
            self.mods = [fitu(self.params,self.u[:,k],mdl=SVR,
                         epsilon=eps,gamma=gamma)
                         for k in range(0,K)]
        
        # Model coefs needed for calcuating lineshape
        self.sv_mat, self.alpha, self.alpha0 = self.model_coefs()

        self.beta0 = np.dot(sv,self.alpha0.transpose())
        self.beta = sv.dot(self.alpha.T)
        self.sigma = 2*gamma*self.sv_mat
        self.sigmabar = np.array(gamma*np.sum(self.sv_mat**2,1))
        self.sigmabar = self.sigmabar.reshape((self.sigmabar.size,1))
    
    def predict(self,phi,unfold=False):
        delta = np.sum(phi**2,0)
        expon = self.sigma.dot(phi) - self.gamma * delta
        expnn = np.exp(expon-self.sigmabar)
        ls = self.beta0 + self.beta.dot(expnn)
        return ls

    def model_coefs(self):
        K = len(self.mods)
        supts = [mod.support_ for mod in self.mods]
        all_supts = np.concatenate(supts)
        all_supts = np.unique(all_supts)
        mem_supts = [np.isin(all_supts,s,assume_unique=True) for s in supts]

        alphas = [mod.dual_coef_ for mod in self.mods]
        sv_mat = np.array(self.params.loc[all_supts,:])

        alpha = np.zeros((sv_mat.shape[0],K))
        for k in range(0,K):
            alpha[mem_supts[k],k] = alphas[k]
        
            alpha0 = np.array([mod.intercept_ for mod in self.mods])
            alpha0 = alpha0.reshape(1,-1)

        return sv_mat, alpha, alpha0

    def save_fit(self,fname='fit',type='csv'):
        if type == "csv":
            np.savetxt(fname+'_beta0'+'.csv',np.array(self.beta0),delimiter=',')
            np.savetxt(fname+'_beta'+'.csv',np.array(self.beta),delimiter=',')
            np.savetxt(fname+'_sigma'+'.csv',np.array(self.sigma),delimiter=',')
            np.savetxt(fname+'_sigmabar'+'.csv',np.array(self.sigmabar),delimiter=',')
            np.savetxt(fname+'_gamma'+'.csv',np.array([self.gamma]),delimiter=',')
            np.savetxt(fname+'_x'+'.csv',np.array([self.x]),delimiter=',')
        elif type == "mat":
            d = {'beta0':self.beta0,
                 'beta':self.beta,
                 'sigma':self.sigma,
                 'sigmabar':self.sigmabar,
                 'gamma':self.gamma,
                 'x':self.x
            }
            sio.savemat(fname+".mat",d)
            

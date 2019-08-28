"""
Example code of how to use SVSA in python. 
"""
import numpy as np
import matplotlib.pyplot as plt
import svsa

train_lineshapes, train_params, xs = svsa.gen_spectra(y_seq=np.linspace(0,2,num=5),
                                                            ri_seq=np.linspace(1.5,3,num=2),
                                                            ef_seq=np.linspace(1.8,2,num=5),
                                                            ci_seq=np.array([1]),
                                                            ct_seq=np.array([1.5]),
                                                            x_seq=np.linspace(-3,3,num=100))
                                                
                                                

svs = svsa.SupportVectorSpectrum(train_params=train_params,train_lineshapes=train_lineshapes,train_x=xs)
svs.fit(nu=1E-1,K=3)
svs.save_fit(fname="test_fit",type="mat") # for use in matlab
svs.save_fit("test_fit",type='csv') # for use in other languages, e.g. R

test_params = np.array([[.8,2,1.9,1,1.5]]).T
pred_line  = svs.predict(test_params)
tenti_line = svsa.crbs6(y=test_params[0],
                           rlx_int=test_params[1],
                           eukenf=test_params[2],
                           c_int=test_params[3],
                           c_tr=test_params[4],
                           xi=np.linspace(-3,3,num=100))['sptsig']

plt.figure(figsize=(20,10))
plt.plot(tenti_line,color='red')
plt.plot(pred_line,color='blue')
plt.show()

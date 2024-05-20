# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # How to use SVSA to learn and predict lineshapes

# First we import the package as well as numpy

import svsa
import numpy as np

# In order to train the SVSA model we need to have some training data. We'll use the built-in Tenti S6 model to generate RB scattering lineshapes.

train_lineshapes, train_params, xs = svsa.gen_spectra(y_seq=np.linspace(0,2,num=5),
                                                            ri_seq=np.linspace(1.5,3,num=2),
                                                            ef_seq=np.linspace(1.8,2,num=5),
                                                            ci_seq=np.array([1]),
                                                            ct_seq=np.array([1.5]),
                                                            x_seq=np.linspace(-3,3,num=100))

# this generates an array of training lineshapes for the given (50) parameters over the (100) values of x

print(train_params.shape)
train_params[0:5] # first five training parameters

print(xs.shape)
xs[0:5] # first five x values used to train

print(train_lineshapes.shape)
train_lineshapes[0:5] # first five lineshapes 

# We can then train SVSA by passing these into the method and calling the fit function (with appropriate choices of nu and K)

svs = svsa.SupportVectorSpectrum(train_params=train_params,train_lineshapes=train_lineshapes,train_x=xs)
svs.fit(nu=1E-1,K=3)

# We can then predict a new line using the predict function passing in the values for the parameters at which we would like to predict the line

test_params = np.array([[.8,2,1.9,1,1.5]]).T
pred = svs.predict(test_params)

# we can then plot and compare it to Tenti's S6 model

import pandas as pd
from scipy.special import wofz

import matplotlib.pyplot as plt
tenti_line = svsa.crbs6(y=test_params[0,0],
                           rlx_int=test_params[1],
                           eukenf=test_params[2],
                           c_int=test_params[3],
                           c_tr=test_params[4],
                           xi=np.linspace(-3,3,num=100))['sptsig']
plt.plot(tenti_line,color='red')
plt.plot(pred,color='blue')
plt.show()

# we can also save the output from the fit coefficients as either a .mat file for MATLAB or as a collection of .csv files for other uses (like R)

svs.save_fit(fname="test_fit",type="mat") # for use in matlab
svs.save_fit("test_fit",type='csv') # for use in other languages, e.g. R

import numpy as np
import pandas as pd
from scipy.special import wofz

def crbs6(y,rlx_int,eukenf,c_int,c_tr,xi):
    n_xi=len(xi)

    n=6
    a=np.zeros((n,n),dtype="complex")
    b=np.zeros((n,2),dtype="complex")

    cohsig=np.zeros((1,n_xi),dtype="complex")
    sptsig=np.zeros((1,n_xi),dtype="complex")

    gamma_int=c_int/(c_tr+c_int)
    j020=-y
    j100=-gamma_int*y/rlx_int
    j001=j100*c_tr/c_int
    j100001=j100*np.sqrt(c_tr/c_int)
    j110=j100*5/6+j020*2/3
    j011110=j100*np.sqrt(5/(8*c_int))
    j_nu=0.4*(1.5+c_int)+(3+c_int)/(2*rlx_int)+9*eukenf/(16*rlx_int**2)
    j_de=-1+(4/15)*eukenf*(1.5+c_int)+(c_int/3)*eukenf/rlx_int
    j_co=-y*(2*gamma_int/3)
    j011=j_co*j_nu/j_de

    def crbs6_solver(xi):
        z=xi+y*1j
        w0=wofz(z)
        if not np.all(np.real(z) == 0):
            w0 = - w0 * 1j * np.pi
            
        w1=-np.sqrt(np.pi)+z*w0
        w2=z*w1
        w3=-0.5*np.sqrt(np.pi)+z*w2
        w4=z*w3
        w5=-3*np.sqrt(np.pi)/4+z*w4
        w6=z*w5
        
        i0000=w0/(np.sqrt(np.pi))
        i0100=(z*w0-np.sqrt(np.pi))*np.sqrt(2/np.pi)
        i0001=i0100
        i0010=(2*w2-w0)/(np.sqrt(6*np.pi))
        i1000=i0010
        i0011=(2*w3-3*w1)/(np.sqrt(5*np.pi))
        i1100=i0011
        i0101=2*w2/np.sqrt(np.pi)
        i0110=(-w1+2*w3)/np.sqrt(3*np.pi)
        i1001=i0110
        i0111=(-3*w2+2*w4)*np.sqrt(2/(5*np.pi))
        i1101=i0111
        i1111=(13*w2-12*w4+4*w6)/(5*np.sqrt(np.pi))

        i1010=(5*w0-4*w2+4*w4)/(6*np.sqrt(np.pi))
        i1110=(7*w1-8*w3+4*w5)/np.sqrt(30*np.pi)
        i1011=i1110
    
        a[:,0]=-j020*np.array([i0000, i0001, i0011, i0010, 0, 0])+np.array([1j, 0, 0, 0, 0, 0])
        a[:,1]=-j020*np.array([i0100, i0101, i0111, i0110, 0, 0])+np.array([0, 1j, 0, 0, 0, 0])
        a[:,2]=(j020-j110)*np.array([i1100, i1101, i1111, i1110, 0, 0])+j011110*np.array([0, 0, 0, 0, -i0100, -i0101])+np.array([0, 0, -1j, 0, 0, 0])
        a[:,3]=(j020-j100)*np.array([i1000, i1001, i1011, i1010, 0, 0])+j100001*np.array([0, 0, 0, 0, -i0000, -i0001])+np.array([0, 0, 0, -1j, 0, 0])
        a[:,4]=j100001*np.array([i1000, i1001, i1011, i1010, 0, 0])+(j001-j020)*np.array([0, 0, 0, 0, i0000, i0001])+np.array([0, 0, 0, 0, 1j, 0])
        a[:,5]=j011110*np.array([i1100, i1101, i1111, i1110, 0, 0])+(j011-j020)*np.array([0, 0, 0, 0, i0100, i0101])+np.array([0, 0, 0, 0, 0, 1j])
    
        b[:,0]=-np.array([i0100, i0101, i0111, i0110, 0, 0])
        b[:,1]=-np.array([i0000, i0001, i0011, i0010, 0, 0])
    
        c=np.linalg.solve(a,b)

        cohsig=np.real(c[0,0]*c[0,0].conjugate())
        sptsig=2*np.real(c[0,1])

        return cohsig, sptsig

    xi = np.array(xi)
    pdxi = pd.DataFrame(xi,columns=["xi"])
    pdxi['xi'].apply(crbs6_solver)
    pdxi['cs']= pdxi["xi"].apply(crbs6_solver)

    cohspt = pd.DataFrame(pdxi['cs'].values.tolist(),columns=["cohsig","sptsig"])
    
    retd = {
        'y':y,
        'rlx_int':rlx_int,
        'eukenf':eukenf,
        'c_int':c_int,
        'c_tr':c_tr,
        'xi':xi,
        'cohsig':cohspt['cohsig'],
        'sptsig':cohspt['sptsig']
        }

    return retd



# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 22:24:23 2023

@author: stefa
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import diff

# function that returns dz/dt
def model_3b(concs,t,k):
    
    k1, k2, k3, k4, k5, k6, k7, k8 = k
    cI, rPoZ, P_cI, cI_dna, A, A_dna = concs
    
    dcI_dt = -cI*(k1*rPoZ + k3*P_cI) + (k2*A + k4*cI_dna) # Unimolecular mass balance
    
    drpoz_dt = -rPoZ*(k1*cI + k5*cI_dna) + (k2*A + k6*A_dna) # Unimolecular mass balance
    
    dcIdna_dt = -k5*cI_dna*rPoZ + k6*A_dna - k4*cI_dna + k3*cI*P_cI 
    
    dPcI_dt = -P_cI*(k3*cI + k7*A) + k4*cI_dna  + k8*A_dna
        
    dA_dt = -k7*A*P_cI + k8*A_dna - k2*A + k1*cI*rPoZ
    
    dAdna_dt = k7*A*P_cI + k5*cI_dna*rPoZ - k8*A_dna - k6*A_dna
    
    dzdt = [dcI_dt, drpoz_dt, dPcI_dt, dcIdna_dt, dA_dt, dAdna_dt]
    return dzdt


def model_4b(concs,t,k):
    
    k1, k2, k3, k4, k5, k6, k7, k8 = k
    cI, rPoZ, P_cI, cI_dna, A, A_dna = concs
    
    dcI_dt = -cI*(k1*rPoZ + k3*P_cI) + (k2*A + k4*cI_dna) # Unimolecular mass balance
    
    drpoz_dt = -rPoZ*(k1*cI + k5*cI_dna) + (k2*A + k6*A_dna) # Unimolecular mass balance
    
    dcIdna_dt = -k5*cI_dna*rPoZ + k6*A_dna - k4*cI_dna + k3*cI*P_cI 
    
    dPcI_dt = -P_cI*(k3*cI + k7*A) + k4*cI_dna  + k8*A_dna
        
    dA_dt = -k7*A*P_cI + k8*A_dna - k2*A + k1*cI*rPoZ
    
    dAdna_dt = k7*A*P_cI + k5*cI_dna*rPoZ - k8*A_dna - k6*A_dna
    
    dzdt = [dcI_dt, drpoz_dt, dPcI_dt, dcIdna_dt, dA_dt, dAdna_dt]
    return dzdt


# initial condition
concs = [1, 1, 0.1, 0, 0, 0]
# [cI, rPoZ, P_cI, cI_dna, A, A_dna]

k1 = 1 # rPoZ + cI binding
k2 = 1 # rPoZ + cI disocciation

k3 = 1 # cI + DNA binding
k4 = 1 # cI + DNA disocciation

k5 = k1 # rPoZ + cI binding w/ cI on DNA
k6 = k2 # rPoZ + cI disocciation w/ cI on DNA

k7 = k3 # cI + DNA binding w/ rPoZ on cI
k8 = k4 # cI +  DNA disocciation w/ rPoZ on cI


k = [k1, k2, k3, k4, k5, k6, k7, k8]

# time points
t = np.linspace(0, 500, num = 100000)

# solve ODE
#z, o = odeint(model,concs,t,args=(k,), full_output = 1, hmin = 0.000001)

# [cI, rPoZ, P_cI, cI_dna, A, A_dna]

def find_eq(concs, k):
    z, o = odeint(model_3b,concs,t,args=(k,), full_output = 1, hmin = 0.000001)
    return z[:,5][99998]

vals = range(1,200)
k1_array = [[x,3,3,1,1,10,10,0.001] for x in vals]


eq_array = []
for i in k1_array:
    eq = find_eq(concs, i)
    eq_array.append(eq)


plt.plot(vals, eq_array)

print(eq_array)
eq = find_eq(concs, k)
#print(eq)
z = []
plotting = False


if plotting:
    plt.figure(dpi=300)
    plt.plot(t,z[:,0], label = "cI")
    plt.plot(t,z[:,1], label = "rPoZ")
    plt.plot(t,z[:,2], label = "Promoter")
    plt.plot(t,z[:,3], label = "cI + promoter complex")
    plt.plot(t,z[:,4], label = "Immunogenic complex")
    plt.plot(t,z[:,5], label = "3-body complex")
    plt.legend()
    plt.show()


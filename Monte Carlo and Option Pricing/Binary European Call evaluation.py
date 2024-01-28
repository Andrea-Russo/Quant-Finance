# -*- coding: utf-8 -*-
"""
Evaluation of European and American call by binary option method
"""

import numpy as np
import numpy.random as rnd

def simulate_asset_price(S0,u,v,T,step_size,num_simulations):
    """"This function simulates the asset price given the initial price,
        the up and down moves in percentage, the Time to maturity
        in years and the step size in months"""
    n=int(12/step_size)
    S=np.zeros([n+1,n+1])
    S[0,0]=S0
    for j in range(n):
        for i in range(n):
            if S[i,j+1]==0:
                S[i,j+1]=S[i,j]*(1+u)
            S[i+1,j+1]=S[i,j]*(1-v)
    return S

def call_option_price(S,K,r):
    """ This function evaluates the payoff at maturity,
        finds the delta hedging at the previous step and propagates
        the option value discounted by the interest rate"""
    s=S.shape
    V=np.zeros(s)
    delta=np.zeros([s[0]-1,s[1]-1])
    
    # Compute payoff at maturity for terminal asset price
    V[:,-1]=np.maximum(S[:,-1]-K,0)
    
    # Compute delta hedge
    for j in range(s[1]-1,0,-1):
        for i in range(s[0]-1,0,-1):
            if S[i,j-1]-S[i,j]!=0:
                delta[i-1,j-1]=(V[i,j-1]-V[i,j])/(S[i,j-1]-S[i,j])
            else:
                delta[i-1,j-1]=0
    
    print(S) 
    print(V)
    print(delta)
    return 
    
S0, u,v=100, 0.03,0.03
T,step_size,num_simulations= 1,4,1000
K,r= 101, 0.01
S=simulate_asset_price(S0, u, v, T, step_size, num_simulations)
call_option_price(S,K,r)

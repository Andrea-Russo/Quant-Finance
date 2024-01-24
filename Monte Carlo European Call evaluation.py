# -*- coding: utf-8 -*-
"""
Evaluation of European call option by Monte Carlo simulation
This program solves the Black-Scholes equation by numerical methods

The steps are the following:
    1-Specify all the free parameters in the option and in the stock price dynamics
    2-Evaluate the option payoff for a randomly generated set of asset values
    3-Discout the payoffs to the present time
    4-Take the expectation value by averaging over the option values

"""
# Import necessary libraries
import numpy as np
import numpy.random as rnd
import datetime

def simulate_asset_price(S0, r, sigma, T, num_simulations):
    """Simulate asset price at maturity. 
    For simple processes where the SDE does not need to be approximated
    like in the case of Geometric Brownian Motion used for calculating
    a European Option Price, we can just simulate the variables at the
    final Time Step as Brownian Motion scales with time and independent
    increments."""
    W = rnd.normal(0,1,num_simulations)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T+sigma*np.sqrt(T)*W)
    return ST

def call_option_price(ST, K, r, T):
    """Calculate the price of a call option."""
    payoff = np.maximum(ST-K, 0)
    V = np.mean(np.exp(-r*T) * payoff)
    return V

# Define stock and option parameters
sigma, S0 = 0.2, 100  # stock parameters
T = ((datetime.date(2022,3,17)-datetime.date(2022,1,17)).days+1)/365  
K, r = 105, 0.01  # option parameters

num_simulations = 100000
ST = simulate_asset_price(S0, r, sigma, T, num_simulations)
V = np.round(call_option_price(ST, K, r, T),2)

print('Value of the call option: ', V)

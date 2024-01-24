# -*- coding: utf-8 -*-
"""
Evaluation of American call
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def simulate_asset_price(S0, r, sigma, T, dt, num_simulations):
    """Simulate asset price paths using Geometric Brownian Motion."""
    num_steps = int(T / dt)
    W = np.random.normal(0, 1, size=(num_steps, num_simulations))
    time_grid = np.arange(1, num_steps + 1) * dt
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W, axis=0))
    S = np.vstack((np.full(num_simulations, S0), S))  # Add S0 to the beginning
    return S, time_grid

def american_option_price(S, K, r, T, dt):
    """Calculate the price of an American call option using the LSM method."""
    num_steps, num_simulations = S.shape[0] - 1, S.shape[1]
    discount_factor = np.exp(-r * dt)
    payoff = np.maximum(S-K, 0)

    # Initialize value function and continuation value
    V = payoff[-1]
    continuation_value = np.zeros_like(V)

    for t in range(num_steps - 1, 0, -1):
        # Use linear regression to approximate continuation value
        in_the_money = S[t] > K
        if np.sum(in_the_money) > 0:
            regression = LinearRegression().fit(S[t, in_the_money].reshape(-1, 1), V[in_the_money])
            continuation_value[in_the_money] = regression.predict(S[t, in_the_money].reshape(-1, 1))

        # Compare continuation value with immediate exercise value
        V = np.where(payoff[t] > continuation_value, payoff[t], V * discount_factor)

    # Discount back one more step to get the option price
    V = V * discount_factor
    return np.mean(V)

# Define stock and option parameters
sigma, S0 =  0.2, 100  # stock parameters
T, K, r = 1, 105, 0.01  # option parameters
dt = 1/252  # daily steps
num_simulations = 100000

S, time_grid = simulate_asset_price(S0, r, sigma, T, dt, num_simulations)
V = american_option_price(S, K, r, T, dt)

print('Value of the American call option: ', V)

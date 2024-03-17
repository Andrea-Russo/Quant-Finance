"""
In this simple program, we simulate the GMB of a stock price over a period of time.
S(t) = S0 * exp((\mu - \sigma^2/2)t + \sigma*\sqrt(t)*W)
"""

# Numerics
import numpy as np
import numpy.random as rnd

# Plotting
import matplotlib.pyplot as plt

print('This program simulates the GBM of a stock with parametrs specified by the user.')
print()

# Geometric Brownian motion parametrs
mu = float(input('Insert stock drift: '))           # drift
vol = float(input('Insert stock volatility: '))     # volatility
S0 = float(input('Inseert stock initial price: '))        # Initial price
T = float(input('Insert time of simulation in years: '))   # Time horizon, in years

# Number of different simulations
iterations = 100

def MonteCarlo(mu, vol, S0, T, iterations):
    # The number of steps in the simulation is obtained to match the numbe of days
    steps = int(T * 365)
    dt = T / steps
    t = np.linspace(0, steps, num=steps)
    
    # Generate random numbers
    W = rnd.randn(iterations, steps)
    
    # Simulate stock price
    S = np.empty((iterations, steps))
    
    # The sumlations are computed in a vectorised way
    for i in range(iterations):
        S[i][0] = S0
        S[i][1:] = S0 * np.exp(np.cumsum((mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * W[i])[1:])
    return S, t

S, t = MonteCarlo(mu, vol, S0, T, iterations)

plt.figure(figsize=(16,7))
for i in range(iterations):
    plt.plot(t,S[i])

plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f"Price of Stock after {T:.2f} years")
plt.grid(True)
plt.show()
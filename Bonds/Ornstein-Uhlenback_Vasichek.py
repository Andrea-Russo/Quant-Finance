"""
Ornsteinâ€“Uhlenbeck process and the Vasichek model of interest rates

In mathematics, the Ornsteinâ€“Uhlenbeck process is a stochastic process. 
Its original application in physics was as a model for the velocity of a massive
Brownian particle under the influence of friction.

The Ornsteinâ€“Uhlenbeck process is a STATIONARY Gaussâ€“Markov process,
which means that it is a GAUSSIAN process, a MARKOV process, and is temporally homogeneous.
In fact, it is the ONLY nontrivial process that satisfies these three conditions,
up to allowing linear transformations of the space and time variables. 
 
Over time, the process tends to drift towards its mean function: 
such a process is called mean-reverting.

The process can be considered to be a modification of the Wiener process, 
in which the properties of the process have been changed so that there is a tendency
of the walk to move back towards a central location, 
with a greater attraction when the process is further away from the center.

The Ornsteinâ€“Uhlenbeck process can also be considered as the continuous-time 
analogue of the discrete-time AR(1) process
"""

import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt


#%% Monte Carlo simulation of Ornstein-Uhlenback process

# Ornstein-Uhlenback parameters
mu = 0.02
theta = 0.01
vol = 0.05
x0 = 0.06
T = 2

# Number of different simulations
iterations = 1000

"""
The process SDE has a formal solution which we can then use to write the simulation
by computing the variance and realising that a temporally homogeneous Ornstein–Uhlenbeck
process can be represented as a scaled, time-transformed Wiener process.


"""


def Ornst_Uhl_Monte_Carlo(mu, theta, vol, x0, T, iterations):
    # Compute time stes
    steps = int(T * 365)
    dt = T / steps 
    t = np.linspace(0, steps, num=steps)
    
    # Generate random numbers
    W = rnd.randn(iterations, steps)
    
    # Simulate process
    x = np.empty((iterations, steps))
    
    # Vectorised simulation using the formal solution and directly utilizing W
    for i in range(iterations):
        x[i][0] = x0
        x[i][1:] = x0 * np.exp(-theta * t[1:]) + mu * (1 - np.exp(-theta * t[1:])) \
                   + vol * np.sqrt((1 - np.exp(-2 * theta * t[1:])) / (2 * theta)) * W[i][1:]
    
    return x, t

# We use r as it will be later considered an interest rate    
r, t = Ornst_Uhl_Monte_Carlo(mu, theta, vol, x0, T, iterations)

plt.figure(figsize=(16,7))
for i in range(iterations):
    plt.plot(t,r[i])

plt.xlabel('Time in days')
plt.ylabel('Value')
plt.title(f"Value of process after {T:.2f} years")
plt.grid(True)
plt.show()


#%%
"""
In finance, the Vasicek model is a mathematical model describing the evolution of interest rates. 
It is a type of one-factor short-rate model as it describes interest rate movements as driven by only one source of market risk. 
The model can be used in the valuation of interest rate derivatives, and has also been adapted for credit markets.

Vasicek's model was the first one to capture mean reversion, an essential characteristic
 of the interest rate that sets it apart from other financial prices. 
Thus, as opposed to stock prices for instance, interest rates cannot rise indefinitely.
This is because at very high levels they would hamper economic activity, prompting a decrease in interest rates. 
Similarly, interest rates do not usually decrease below 0. 
As a result, interest rates move in a limited range, showing a tendency to revert to a long run value.

The main disadvantage is that, under Vasicek's model, it is theoretically possible for the interest rate to become negative,
an undesirable feature under pre-crisis assumptions. 
This shortcoming was fixed in the Cox–Ingersoll–Ross model, exponential Vasicek model, 
Black–Derman–Toy model and Black–Karasinski model, among many others. 
The Vasicek model was further extended in the Hull–White model. 
he Vasicek model is also a canonical example of the affine term structure model, along with the Cox–Ingersoll–Ross model. 
In recent research both models were used for data partitioning and forecasting.
"""

"""
Under the no-arbitrage assumption, a discount bond (a.k.a. zero coupon) may be priced in the Vasicek model. 
The time t value of a discount bond with maturity date T is exponential affine in the interest rate.
Here, let, t=0 such that T=tau is the time to maturity and P the bond price, then
"""


# Now derive price by using simulated O-U process
F = 1000 # Prinicipal value
Price = F * np.exp(-r)

# Compute expected price averaging over columns
avg_price = np.mean(Price, axis=0) 

"""
If one desires to plot all the simulated prices, one can uncomment this
# Plot prices
plt.figure(figsize=(16,7))
for i in range(iterations):
    plt.plot(t,Price[i])

plt.xlabel('Time in days')
plt.ylabel('Bond Price')
plt.title(f"0-coupon Bond prices with coupon at {T:.2f} years")
plt.grid(True)
plt.show()
"""

# Plot average price
plt.figure(figsize=(16,7))
plt.plot(t,avg_price)
plt.xlabel('Time in days')
plt.ylabel('Average Bond Price')
plt.title(f"Avg 0-coupon Bond price with coupon at {T:.2f} years")
plt.grid(True)
plt.show()


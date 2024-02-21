"""
Variance reduction with Antithetic Variates.

In this program we show how to have a more precise Monte Carlo option pricing 
by using antithetic variates, a.k.a. simulating a stock price that is perfectly negative correlated.
Much like, hedging reuced the variance of a portfolio through negative correlation,
anthitetic variates reduce the variance of the Montecarlo simulation.

If we are trying to estimate the variable Y and we simulate using the random variable X, 
the estimate will have Var(Y) = Var(X).
However, if we extimate by Y = (X1 + X2)/2, Var(Y) = (Var(X1) + Var(X2) + 2COV(X1,X2)) /4.
If X1 and X2 are negatively correlated, the variance will be smaller than just using a single variable.

In particular, if X2=-X1, then <Y>=0 and Var(Y)= 0.5*Var(X1) - 0.5*Var(X1) = 0
"""
# Numerics
import numpy as np
import numpy.random as rnd
import scipy.stats as stats

# Plotting
import matplotlib.pyplot as plt

#%% Define functions to simulate Monte carlo and compute option price.

def anthitetic_MC(r, vol, T, S0, N):
    W = rnd.randn(N)
    S = S0 * np.exp((r - vol**2 / 2)*T + vol*np.sqrt(T)*W)
    S2 = S0 * np.exp((r - vol**2 / 2)*T - vol*np.sqrt(T)*W)
    return S, S2

def callPricer(r, K, S, S2, T, N):
    # Pricing without antitheti variate
    payoff = np.maximum(0, S-K)
    V = np.exp(-r*T) * np.mean(payoff)
    sigma = np.sqrt(np.sum( (payoff - V )**2 ) / (N-1))
    SE = sigma / np.sqrt(N)
    
    
    # Pricing with antitheti variate
    payoff_ant = 1/2 * (np.maximum(0, S-K) + np.maximum(0, S2-K))
    V_ant = np.exp(-r*T) * np.mean(payoff_ant)
    sigma_ant = np.sqrt(np.sum( (payoff_ant - V_ant )**2 ) / (N-1))
    SE_ant = sigma_ant / np.sqrt(N)
    return V, SE, V_ant, SE_ant

#%% Request necessary parameters from user and compute option prices and standard error.

print('This program computes the prices of a EU call option using Monte Carlo simulations.')
print('The program shows how the use of Antithetic Variates improves Monte Carlo convergence.')
S0 = float(input('Insert currrent stock price: '))
K = float(input('Insert option strike price: '))
r = float(input('Insert Interest rate: '))
T = int(input('Insert days to expiry: ')) / 365
vol = float(input('Insert implied volatility: '))


N = 10000
S, S2 = anthitetic_MC(r, vol, T, S0, N)

V, SE, V_ant, SE_ant = callPricer(r, K, S, S2, T, N) 


print(f"\nThe call price is ${V:.2f} with standard error +/- ${SE:.3f}")
print(f"The call price with Antithetic Variate is ${V_ant:.2f} with standard error +/- ${SE_ant:.3f}")

#%% Visualisation of convergence


# Set up PDFs
x_range = np.linspace(min(V-3*SE, V_ant-3*SE_ant), max(V+3*SE, V_ant+3*SE_ant), 1000)
pdf_standard = stats.norm.pdf(x_range, V, SE)
pdf_antithetic = stats.norm.pdf(x_range, V_ant, SE_ant)

# Plotting
plt.figure(figsize=(16, 8))

# Antithetic Variate method regions
plt.fill_between(x_range, pdf_antithetic, where=(x_range < V_ant - SE_ant), color='blue', alpha=0.5, label='More than 1 Std Dev (Antithetic)')
plt.fill_between(x_range, pdf_antithetic, where=((x_range >= V_ant - SE_ant) & (x_range <= V_ant + SE_ant)), color='cornflowerblue', alpha=0.5, label='Within 1 Std Dev (Antithetic)')
plt.fill_between(x_range, pdf_antithetic, where=(x_range > V_ant + SE_ant), color='blue', alpha=0.5)

# Standard Monte Carlo method
plt.plot(x_range, pdf_standard, label='Standard Monte Carlo Method', color='red')

# Theoretical values
plt.axvline(V, color='orange', linestyle='--', label='Theoretical Value (No Antithetic)')
plt.axvline(V_ant, color='black', linestyle='--', label='Theoretical Value (Antithetic)')

# Adjust y-axis limits to better visualize the distribution
plt.ylim(0, max(max(pdf_standard), max(pdf_antithetic)) * 1.1)

plt.title("Normalized Probability Distributions of Estimated Call Prices")
plt.xlabel("Option Price")
plt.ylabel("Probability Density")
plt.legend()
plt.show()


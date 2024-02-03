"""
MEAN-VARIANCE PORTFOLIO THEORY

MVP Theory is based on the premise that returns are normally distributed 
and by looking at mean and variance, we can describe the distribution of 
end-of-period wealth.

The basic idea of this theory is to achieve diversification by constructing 
a portfolio for a minimal portfolio risk or maximal portfolio returns given 
a certain level of risk. Accordingly, the Efficient Frontier is a set of 
optimal portfolios in the risk-return spectrum, and portfolios located under 
the Efficient Frontier curve are considered sub-optimal.

This means that the portfolios on the frontier offer:
-Highest expected return for a given level of risk
-Lowest level of risk for a given level of expected returns

In this program we obtain the efficient portfolio frontier and identity the weights
for different portfolio allocations, like the minimum variance portfolio, with 
different combinations of risky and riskless assets.
"""

#%% Import libraries

# Numerics
import numpy as np
import numpy.random as rnd

# Import optimization module from scipy
import scipy.optimize as sco

# Data
import pandas as pd
import yfinance as yf

# Plotting
import matplotlib.pyplot as plt

#%% Retrieve data

# We will use chips stocks to construct our porfolio
tickers = ['KWEB', 'VZ', 'NEM', 'TSN', 'SBUX']

num_asset = len(tickers)
num_simul = 20000

# Fetching data from yfinance (disable progeress bar)
start_date = '2015-01-01'
end_date = '2024-01-01'
stocks = yf.download(tickers, start=start_date, 
                          end=end_date, progress=True)['Adj Close']

# Save data
stocks.to_csv('stocks.csv')

# Load locally stored data
df = pd.read_csv('stocks.csv', index_col=0, parse_dates=True)

#%% Exploratory data analysis

# Statistical Summary
summary = df.describe().T
print(summary)

# Visualisation of last trading year
fig = plt.figure(figsize=(16,8))
ax = plt.axes()

# Choose how many days prior to the latest date the plot shows:
plot_days = 250
"""
Normalizes these rows by dividing them by the first row in this subset 
and then multiplies by 100 to convert to a percentage. 
This normalization process adjusts all series in the DataFrame so they start 
at the same point (100%), making it easier to compare their relative performance
over the time period 
"""
ax.plot(df[-plot_days:]/df.iloc[-plot_days] * 100)
ax.set_title('Normalized Price Plot')
ax.legend(df.columns, loc='upper left')
ax.grid(True)
ax.set_ylabel('Percentage (%)')
ax.set_xlabel(f"Trading Days (Last {plot_days} Days)")
plt.ylabel('Percent Change from Starting Point')

plt.show()

#%% Compute returns

# Comnpute pct returns
returns = df.pct_change().fillna(0)

# Compute log returns
log_returns = np.log(df / df.shift(1)).fillna(0)

# Annualise returns
annual_returns = returns.mean() * 252
annual_log_returns = log_returns.mean() * 252

print(f"Annualised percentage returns: {annual_returns}")
print(f"Annualised log-returns: {annual_log_returns}")

fig = plt.figure()
ax = plt.axes()
ax.bar(annual_log_returns.index, annual_log_returns*100, color='r', alpha=0.8)
ax.set_title('Annualised log-returns (in %)')

""" we proceed using log-returns"""

#%% Compute historical volatility
vols = log_returns.std()

# Calculate annualized volatilities
annual_vols = vols * np.sqrt(252)

# Visualize the data
fig = plt.figure()
ax = plt.axes()

ax.bar(annual_vols.index, annual_vols*100, color='orange', alpha=0.5)
ax.set_title('Annualized Volatility (in %)');

#%% Portfolio statistics

"""
Consider constructing a fully invested portfolio in these assets
"""

# Portfolio returns
ret = np.array(log_returns.mean()*252)[:,np.newaxis]
print(f"Portfolio resturns: {ret}")

# Portfolio covariance matrix
annual_covar = log_returns.cov() * 252

# Portfolio statistics function
def portfolio_stats(weights, annualised_returns, annualised_covariance):
    """
    Given the weights of the assets in the portfolio,
    this function computes the portfolio statistics.
    """
    # Convert weights and annualised_returns to numpy arrays without adding unnecessary dimensions
    weights = np.array(weights).reshape(-1, 1)  # Ensuring weights is a column vector
    annualised_returns = np.array(annualised_returns)
    
    # Calculate portfolio return
    port_ret = np.dot(weights.T, annualised_returns)
    
    # Calculate portfolio volatility
    # Ensuring dot operations are between correctly aligned matrices/vectors
    port_vol = np.sqrt(np.dot(np.dot(weights.T, annualised_covariance), weights))[0, 0]  # Simplify the result to a scalar
    
    # Calculate Sharpe ratio (assuming the risk-free rate is 0 for simplicity)
    sharpe = port_ret / port_vol
    
    return np.array([port_ret.item(), port_vol, sharpe.item()])  # Use .item() to extract scalar values

#%% Portfolio simulation

"""
We now run a Monte Carlo simulation to generate random portfolio weights
and compute the expected return, variance and Sharpe ratio.
We then use them to analayse and individuate portfolios
"""

# Initialise list
returns_list = []
vol_list = []
weights_list = []
sharpe_list = []

# Simulate portfolios
for i in range(num_simul):
    
    # Generate weights between 0 and 1 and normalise
    weights = rnd.uniform(0, 1, num_asset)[:, np.newaxis]
    weights = weights / sum(weights) 
    
    # Call functio nto generate portfolio statistics
    port_ret, port_vol, sharpe  = portfolio_stats(weights, annual_log_returns, annual_covar)
    
    # Save portfolio stats
    returns_list.append(port_ret)
    vol_list.append(port_vol)
    weights_list.append(weights.flatten())
    
# record values 
portfolio_returns = np.array(returns_list).flatten()
portfolio_vols = np.array(vol_list).flatten()
portfolio_weights = np.array(weights_list)
portfolio_sharpe = portfolio_returns / portfolio_vols


#%% Simulated portfolio analysis

# Subsume results into datframe for analysis

# Convert the list of weights into a DataFrame
weights_col = [f'weight_{i}' for i in range(num_asset)]
weights_df = pd.DataFrame(portfolio_weights, columns=weights_col)

# Combine the other metrics into a DataFrame
metrics_df = pd.DataFrame({
    'log-returns': portfolio_returns,
    'volatility': portfolio_vols,
    'sharpe_ratio': portfolio_sharpe})

# Concatenate the metrics with the weights along the horizontal axis
MVP_df = pd.concat([metrics_df, weights_df], axis=1)

mvp_summary = MVP_df.describe()

# Find the index of the portfolio with the maximum Sharpe ratio
max_sharpe_idx = MVP_df['sharpe_ratio'].idxmax()

# Extract the row corresponding to the maximum Sharpe ratio
max_sharpe = MVP_df.loc[max_sharpe_idx]
print(f"The maximum Sharpe ratio is {max_sharpe['sharpe_ratio']}")

# Extract the portfolio weights for the maximum Sharpe ratio
max_sharpe_port_wts = MVP_df.loc[max_sharpe_idx, weights_col]

# Assuming 'tickers' is a list of asset names corresponding to the weights
max_sharpe_alloc = dict(zip(tickers, np.around(max_sharpe_port_wts.values * 100, 2)))
print(f"The allocation for the maximum Sharpe ratio portfolio is: \n{max_sharpe_alloc}")

#%% Visualise simulated portfolios
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_title('Monte Carlo Simulated Allocation')



# Scatter plot for the portfolios
# Color by Sharpe ratio; this time using actual returns
scatter = ax.scatter(portfolio_vols, portfolio_returns, c=portfolio_sharpe,
                     cmap='RdYlGn', marker='o', edgecolor='k', alpha=0.7)

# Color bar for Sharpe ratio
colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('Sharpe Ratio')

# Highlighting the portfolio with the maximum Sharpe ratio
ax.scatter(max_sharpe['volatility'], max_sharpe['log-returns'],
           color='red', s=100, edgecolors='black', label='Max Sharpe Ratio Portfolio', zorder=5)

# Labels and title
ax.set_title('Simulated Portfolios with Log-Returns', fontsize=14)
ax.set_xlabel('Expected Volatility', fontsize=12)
ax.set_ylabel('Expected Actual Return', fontsize=12)
ax.grid(True)
ax.legend()

#%% Visualise actual returns

# Conversion of annualized log returns to annualized actual returns
annualized_actual_returns = np.exp(portfolio_returns) - 1

# Update the DataFrame to include actual returns
MVP_df['actual_returns'] = annualized_actual_returns

# Plotting the simulations with actual returns
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot for the portfolios
# Color by Sharpe ratio; this time using actual returns
scatter = ax.scatter(MVP_df['volatility'], MVP_df['actual_returns'], 
                     c=MVP_df['sharpe_ratio'], cmap='viridis', marker='o', edgecolor='k', alpha=0.7)

# Color bar for Sharpe ratio
colorbar = fig.colorbar(scatter, ax=ax)
colorbar.set_label('Sharpe Ratio')

# Highlighting the portfolio with the maximum Sharpe ratio
max_sharpe_actual_return = np.exp(max_sharpe['log-returns']) - 1
ax.scatter(max_sharpe['volatility'], max_sharpe_actual_return,
           color='red', s=100, edgecolors='black', label='Max Sharpe Ratio Portfolio', zorder=5)

# Labels and title
ax.set_title('Simulated Portfolios with Actual Returns', fontsize=14)
ax.set_xlabel('Expected Volatility', fontsize=12)
ax.set_ylabel('Expected Actual Return', fontsize=12)
ax.grid(True)
ax.legend()

plt.show()

#%% Max-Sharpe ration prtfolio
"""
We use the scipy optimisation module to find the portfolio 
with the maximum sharpe ratio
"""

# Maximizing sharpe ratio
def min_sharpe_ratio(weights, annual_log_returns, annual_covar):
    return -portfolio_stats(weights, annual_log_returns, annual_covar)[2]

# Constraints to ensure the sum of weights is 1 (fully invested portfolio)
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds to ensure each weight is between 0 and 1
bnds = tuple((0, 1) for _ in range(num_asset))

# Initial weights (equally distributed)
initial_wts = np.array(num_asset * [1. / num_asset])

# Optimizing for maximum Sharpe ratio using Sequential Least Squares Programming (SLSQP)
opt_sharpe = sco.minimize(lambda weights: min_sharpe_ratio(weights, annual_log_returns, annual_covar), 
                          initial_wts, method='SLSQP', bounds=bnds, constraints=cons)


optimized_weights = np.around(opt_sharpe.x * 100, 2)
ticker_weights = list(zip(tickers, optimized_weights))

print("Optimized Portfolio Weights:")
for ticker, weight in ticker_weights:
    print(f"{ticker}: {weight}%")

# Portfolio stats
optimized_portfolio_stats = portfolio_stats(opt_sharpe.x, annual_log_returns, annual_covar)
stats_labels = ['Returns', 'Volatility', 'Sharpe Ratio']
optimized_stats = np.around(optimized_portfolio_stats, 4)

print("Optimized Portfolio Statistics:")
for stat, value in zip(stats_labels, optimized_stats):
    print(f"{stat}: {value}")
    
#%% Min variance portfolio


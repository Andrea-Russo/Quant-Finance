"""
Andrea Russo (andrearusso.quant@gmail.com)

In this program, we find the Implied Volatility of a call option for a ticker decided by the user.

We assume the call option is a European-style option, while in reality it is American.

The option prices for a given time to expiry and strike price can be dowloaded
from the web. However, the volatility of Black-Scholes pricing equations cannot be computed by inverting
the formula, but must be obtained numerically.

We assume a non-dividend paying stock.

In this program we find the exact implied volatility usign the Newton-Rhapson method
"""
#%% Import modules

# Math
import numpy as np
from scipy.stats import norm

# Financial data
import yfinance as yf
from fredapi import Fred
import pandas as pd

# Import time
from datetime import datetime, timedelta

# Plotting
import matplotlib.pyplot as plt

#%% Dowload Option prices

# Define ticker symbol
ticker_symbol = input('Please insert desired ticker: ').upper()  # Google

# Create ticker object
ticker = yf.Ticker(ticker_symbol)

# Get option expiry dates
opt_maturities = ticker.options

# Get today's date
today = datetime.today()

# Time to expiry
# Here we are approximating 3 months as 90 days
T = 90

# Calculate the target date approximately 3 months from today
target_date = today + timedelta(days=T)

# Convert the expiration dates to datetime objects 
# and find the closest to the target date
closest_exp_date = min(opt_maturities, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - target_date))

# Get option chain for selected expiration
option_chain = ticker.option_chain(closest_exp_date)

# The option chain includes both calls and puts
# These are already in Pandas DataFrame format
calls = option_chain.calls
puts = option_chain.puts

# Display calls and puts 
print('Calls data:')
print(calls.head())

#print('Puts data:')
#print(puts.head())

#%% Retrieve interest rates from the Federal reserve Economic Data

#Initialize the Fred object with your API key
fred = Fred(api_key='ff7bcf7eb5e7d58e26b8f24fe31b85fb')

"""
Possible FRED Series IDs:

Interest Rates:
- Federal Funds Effective Rate: 'FEDFUNDS'
- 10-Year Treasury Constant Maturity Rate: 'DGS10'
- 3-Months T-Bills rate: 'DTB3'
- 1-Year Treasury Constant Maturity Rate: 'DGS1'

Inflation:
- Consumer Price Index for All Urban Consumers: All Items: 'CPIAUCSL'
- Personal Consumption Expenditures Excluding Food and Energy (Core PCE): 'PCEPILFE'

Unemployment:
- Civilian Unemployment Rate: 'UNRATE'
- Initial Jobless Claims: 'ICSA'

Gross Domestic Product (GDP):
- Real Gross Domestic Product: 'GDPC1'
- Gross Domestic Product: 'GDP'

Other Economic Indicators:
- Industrial Production Index: 'INDPRO'
- Retail Sales: 'RSXFS'
- Housing Starts: 'HOUST'

"""


# Define the series ID for the interest rate. 
# We choose the 3 motnh rate as that is the time to expiry
series_id = 'DTB3'

# Fetch the data
interest_rate_data = fred.get_series(series_id)

# Display the data
print('Interest rate:')
print(interest_rate_data.tail())  # This prints the most recent data points


#%% Plot volatility graph
"""
At this point, we can pick a strike price K, stock price S and interest rate r.
The Black-Scholes option formula will then give the option price as a function
of volatility,the only free parameter C=C(sigma).
Let C_0 be the option price from the market. 
We can then plot C(sigma)-C_0=0, which is the difference between the theoretical
and market price of the option as a function of the volatility.
"""

# Download stock price
# Here, 'period' is set to '1d' for the most recent day and 'interval' to '1m' to get the latest minute data
latest_price_data = ticker.history(period="1d", interval="1m")
# The 'Close' column of the last row will have the most recent closing price
latest_price = latest_price_data['Close'].iloc[-1]
#print("Latest stock price:", latest_price)

# Set S as the latest market price. 
S = latest_price

# Set K as the strike price. We find the closest strike for S+5
closest_strike_idx = (np.abs(calls['strike'].values - (S+5))).argmin()
K = calls['strike'].iloc[closest_strike_idx]

# Set r as the latest FED funds rate.
r = interest_rate_data.iloc[-1] / 100

# Set time to expiry tau using the date of the fecthed options, matching the options
# The time must be expressed in years
tau = (datetime.strptime(closest_exp_date, "%Y-%m-%d") - today).days / 365

# Generate volatilities from 0 to 1 (percentages)
size = 100
sig = np.linspace(0.01, 1, num=size)

print(f"Stock: ${S}, Strike: ${K}, Int rate: {r}, Time to exp: {tau*365} days.")

# Compute d1
d1 = (1/(np.sqrt(tau) * sig)) * (np.log(S/K) + (r + np.square(sig) / 2) * tau)

# Compute d2
d2 = d1 - sig * np.sqrt(tau)

# Compute Black Scholes price C(sig)
C = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

# Extract market price C_0
C_0 = calls['lastPrice'].iloc[closest_strike_idx]

# Plotting C(sigma) - C_0 against sigma
plt.figure(figsize=(16,8))
plt.plot(sig, C - C_0, label='C($\sigma$) - C_0')
plt.axhline(0, color='black', lw=1, linestyle='--') # Highlight the y=0 axis
plt.title('Black-Scholes Difference from Market as a Function of Volatility $\sigma$', fontsize=14)
plt.xlabel('Volatility ($\sigma$)', fontsize=10)
plt.ylabel('C($\sigma$) - C_0', fontsize=10)
plt.grid(visible=True)
plt.legend()
plt.show()

#%% Now, we use the Newton-Rhapson method with update x=x_0-f(x_0)/f'(x_0)

# Set an error bound
epsilon = 1e-4

# First, we sample an insitial guess of x_0, in this case sig_0
sig_0 = np.random.uniform(0,1)

def NewtonRap(S, K, r, sig_0, tau, epsilon, C_0, max_iter=1000):
    for i in range(max_iter):
        
        # Compute d1 and d2 for this guess
        d1_0 = (1/(np.sqrt(tau) * sig_0)) * (np.log(S/K) + (r + np.square(sig_0) / 2) * tau)
        d2_0 = d1_0 - sig_0 * np.sqrt(tau)
        
        # Compute option price C(sigma) and its difference from market price
        C_sig_0 = S * norm.cdf(d1_0) - K * np.exp(-r * tau) * norm.cdf(d2_0) - C_0

        # Check if the function value is within the error bound
        if abs(C_sig_0) < epsilon:
            return sig_0
        
        # Compute the Vega (derivative of the option price)
        vega = S * norm.pdf(d1_0) * np.sqrt(tau)

        # Avoid division by zero
        if vega == 0:
            break

        # Update sigma using the Newton-Raphson formula
        sig_0 -= C_sig_0/vega

    # In case of non-convergence, return None or an appropriate value
    return None

# Use function to compute implied vol
impl_vol = NewtonRap(S, K, r, sig_0, tau, epsilon, C_0)
if impl_vol is not None:
    impl_vol = round(impl_vol * 100, 2)
    print(f"The implied volatility computed for these data is {impl_vol}%")
else:
    print("Failed to converge to a solution.")

real_vol = round(calls['impliedVolatility'].iloc[closest_strike_idx] * 100,2)
print(f"The implied volatility retrieved from yfinance is {real_vol}%")

print('The discrepancy could be attributed to the limits of Black-Scholes pricing or to the different option nature.')



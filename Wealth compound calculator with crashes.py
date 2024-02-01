"""
Compound interest calculator with crashes.

This program computes the compound return of wealth over the span of T years
and compares it with the wealth obtained if the money is uninvested.

Rate of return is assumed to be constant.
Yearly contributions are optional.
Crashes are optional and the size can be controlled.
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

def get_positive_input(prompt, input_type=float):
    """ Function to get positive input from the user. """
    while True:
        try:
            value = input_type(input(prompt))
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive number.")

# Define parameters
T = get_positive_input('Insert number of years of investing: ', int)
S0 = get_positive_input('Insert initial investment capital: ')

contrib = input("If you intend to contribute yearly type 'YES', otherwise type 'NO': ")
while contrib.upper() not in ['YES', 'NO']:
    print('Input not valid, please try again.')
    contrib = input("If you intend to contribute yearly type 'YES', otherwise type 'NO': ")

yearly = get_positive_input('Insert yearly contribution to the investment: ') if contrib.upper() == 'YES' else 0
r = get_positive_input('Insert yearly % rate of return of the investment: ') / 100

crash_num = get_positive_input('Insert number of crashes: ', int)
crash_years = set()
crashes_size = []

if crash_num > 0:
    for i in range(crash_num):
        time = get_positive_input(f"Insert year of the {i+1} crash: ", int)
        while time in crash_years:
            print('The year has already been inserted, insert a different year: ')
            time = get_positive_input(f"Insert year of the {i+1} crash: ", int)
        crash_years.add(time)
        crashes_size.append(get_positive_input('Insert market crash %: ') / 100)

# Crash years need to be converted to a list before being sorted
crash_years = sorted(list(crash_years))

# Define functions
def compounder(T, S0, yearly, r, crash_years=None, crashes_size=None):
    if not crash_years:
        crash_years = []
    
    S = np.zeros(T)
    S_cash = np.zeros(T)
    S[0] = S0
    S_cash[0] = S0
    
    for i in range(1, T):
        if i in crash_years:
            j = crash_years.index(i)
            S[i] = S[i-1] * (1-crashes_size[j]) + yearly
            S_cash[i] = S_cash[i-1] + yearly
        else:
            S[i] = S[i-1] * (1+r) + yearly
            S_cash[i] = S_cash[i-1] + yearly
    
    return S, S_cash 

# Compute returns
S, S_cash = compounder(T, S0, yearly, r, crash_years, crashes_size)

try:
    ret = (S[-1] - S_cash[-1]) * 100 / S_cash[-1]
    print(f"\nAfter {T} years of investing, the wealth amounts to £{S[-1]:.2f}.")
    print(f"If the money is not invested, the wealth amounts to £{S_cash[-1]:.2f}.")
    print(f"The total return is {ret:.2f}%.")
except ZeroDivisionError:
    print("Error in calculating returns. Please check the inputs.")

# Plot
t = np.arange(0, T, step=1)
plt.figure(figsize=(16, 8))
plt.plot(t, S, '--o', color='Red', label='Invested')
plt.plot(t, S_cash, '--o', color='Blue', label='Not Invested')
plt.title('Wealth as a function of time', fontsize=16)
plt.xlabel('Time in years', fontsize=12)
plt.ylabel('Wealth in £', fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

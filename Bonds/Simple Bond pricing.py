"""
This program computes the present value of bonds. 
Constant interest rates and coupon payments are assumed
"""

def bond_pres_val(face_val, coupon, rate, intervals):
    cashflow = face_val * coupon * sum([(1 + rate)**(-i) for i in range(1,intervals+1)])
    return cashflow + face_val / (1 + rate)**intervals

time_maturity = int(input('Insert years to maturity: '))
face_val = float(input('Insert face value of the bond: '))
freq = int(input('Insert number of coupon payments per year: '))
rate = float(input('Insert annual interest rate: ')) / freq
coupon = float(input('Insert annual coupon rate: ')) / freq

intervals = time_maturity * freq

present_value = round(bond_pres_val(face_val, coupon, rate, intervals),2)
print(f"The present value of the bond is £{present_value}.")

price = float(input('Insert current bond price: '))
bond_yield = round(100 * coupon * face_val * freq / price,2) 
print(f"The yield of this bond is: {bond_yield}%")
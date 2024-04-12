"""
PCA analysis of the yield curve and its factors.
"""

#%% Import libraries

# Numerical
import numpy as np

# Data
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt

#%% Data upload and manipulation

data = pd.read_csv('hjm-pca.txt', index_col=0, sep='\t')

"""
Representation of a yield curve as 50 forward rates. As the yield curve evolves over time, each forward rate can change. 
It is understood that adjacent points on the yield curve do not move independently. 
PCA is a method for identifying the dominant ways in which various points on the yield curve move together.

PCA allows us to take a set of yield curves, process them using standard mathematical methods, 
and then define a reduced form model for the yield curve. 
This reduced form model retains only a small number of principal components (PCs) 
but can reproduce the vast majority of yield curves that the full structural model could. 
This reduced model has fewer sources of uncertainty (i.e. dimensions) than if the 50 points of the yield curve were modelled independently.
"""

#%% Plotting
# Increase the default plot size and set a higher DPI for better quality.
plt.figure(figsize=(12, 4), dpi=120)

# Plotting the first row of your data
plt.plot(data.iloc[0], marker='o', linestyle='-')  # Use a line and markers for visibility

# Adding titles and labels with increased font sizes
plt.title('Representation of a Yield Curve', fontsize=16)
plt.xlabel('Forward Rate Index', fontsize=14)
plt.ylabel('Rate', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.grid(True)

# Options to adjust the x and y limits to zoom in to a specific part of the data if needed
# plt.xlim([x_min, x_max])
# plt.ylim([y_min, y_max])

plt.show()

# All curves
# Transposing data to have each column represent a date
transposed_data = data.T

# Setup a larger figure to handle multiple curves
plt.figure(figsize=(14, 7), dpi=100)

# Plot each day's yield curve. Each column in 'transposed_data' is a separate day's curve.
for column in transposed_data.columns:
    plt.plot(transposed_data.index, transposed_data[column], label=f'Day {column}')

# Adding titles and labels
plt.title('Daily Yield Curves', fontsize=16)
plt.xlabel('Forward Rate Index', fontsize=14)
plt.ylabel('Rate', fontsize=14)

# Improving layout to avoid cutting off labels
plt.tight_layout()

# Rotate x-axis labels if necessary
plt.xticks(rotation=45)

# Adding a grid for better readability
plt.grid(True)

#%% Volatility chart
"""
We'll now produce the volatility chart by taking the first difference (scaling) and calculating historical variance by each individual maturity.
"""

diff_ = data.diff(-1)
diff_.dropna(inplace=True)
vol = np.std(diff_, axis=0) * 10000

vol_data = vol[:21]  # Selecting the first 21 entries

# Creating the plot
plt.figure(figsize=(10, 5))
plt.plot(vol_data, marker='o', linestyle='-', color='cornflowerblue')  # Use 'cornflowerblue' as specified

# Setting titles and labels
plt.title('Volatility of Daily UK Government Yields', fontsize=16)
plt.xlabel('Tenor', fontsize=14)
plt.ylabel('Volatility (bps)', fontsize=14)

# Rotating x-axis labels for better readability if necessary
plt.xticks(rotation=45)


plt.grid(True)
plt.show()

#%% PCA
"""
The volatility plot is of the averaged values, but we can see that different parts of the yield curve move differently. 
Volatility is very significant, especially at the shorter end of the curve. 
This means that 1-year and 2-year rates seems to move up and down a lot as compared to other maturities.

It is never all up or all down and PCA help us figure out exactly what is going. 
Covariance of daily changes shows dependency of different rates. 
Principal components can be calculated by finding the eigenvalues and eigenvectors of this covariance matrix.
"""

cov_= pd.DataFrame(np.cov(diff_, rowvar=False)*252/10000, columns=diff_.columns, index=diff_.columns)

# Now perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_)

# As a good practice, sort values
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

df_eigenval = pd.DataFrame(eigenvalues_sorted, columns=['Eigenvalues'])

# Calculate explained variance proportion
df_eigenval["Explained proportion"] = df_eigenval["Eigenvalues"] / np.sum(df_eigenval["Eigenvalues"])

# Format explained proportion as percentage for display purposes (this will not be used in calculation)
df_eigenval["Explained Proportion %"] = df_eigenval["Explained proportion"].apply(lambda x: f"{x:.2%}")

# Print the DataFrame to view the formatted percentages
print(df_eigenval[['Eigenvalues', 'Explained Proportion %']][:10])

# Calculate percentage for plotting (multiply by 100)
percentage_variance = df_eigenval['Explained proportion'][:10] * 100

# Creating the bar plot
plt.figure(figsize=(10, 6))
plt.bar(percentage_variance.index, percentage_variance, color='cornflowerblue')
plt.title('Percentage of Overall Variance Explained', fontsize=16)
plt.xlabel('Components', fontsize=14)
plt.ylabel('Explained Variance (%)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot of most relevant 3 components
pcadf = pd.DataFrame(eigenvectors[:,0:3], columns=['PC1','PC2','PC3'])

fig, ax1 = plt.subplots(figsize=(10, 4))

# Plotting PC2 and PC3 on the primary y-axis
ax1.plot(pcadf.index, pcadf['PC2'], label='PC2', color='blue', linestyle='-', marker='o')
ax1.plot(pcadf.index, pcadf['PC3'], label='PC3', color='green', linestyle='-', marker='x')

# Setting labels and title
ax1.set_xlabel('Index', fontsize=14)
ax1.set_ylabel('Change in yield (bps)', fontsize=14)
ax1.set_title('First Three Principal Components', fontsize=16)

# Setting up the secondary y-axis for PC1
ax2 = ax1.twinx()
ax2.plot(pcadf.index, pcadf['PC1'], label='PC1', color='red', linestyle='-', marker='s')
ax2.set_ylabel('PC1', fontsize=14)

# Adding legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')
ax1.grid(True)
plt.show()

"""
One of the key interpretations of PCA as applied to interest rates are the components of the yield curve. 
We can attribute the first three principal components to:

-Parallel shifts in yield curve (shifts across the entire yield curve)
-Changes in short/long rates (steepening/flattening of the curve)
-Changes in curvature of the model (twists)

The first PC represents the situation that all forward rates in the yield curve move in the same direction 
but points around the 15 year term move more than points at the shorter or longer parts of the yield curve. 
This corresponds to a general rise (or fall) of all of the forward rates in the yield curve, 
but cannot be called a uniform or parallel shift. 
The impact of the first PC can be easily observed amongst the yield curves as it contributes more than 71% of the variability.

The second PC represents situations in which the short end of the yield curve 
moves up at the same time as the long end moves down, or vice versa. 
This is often described as a tilt in the yield curve, 
although in practice there is more subtle definition to the shape. 
This reflects the particular yield curves that were used for the analysis, 
as well as the structural model and calibration that were used to create them. 
In this example, the influence of the second PC accounts for about 16.27% of the variability in the yield curves.

The third PC is further interpreted as a higher order buckling in which 
the short end and long end move up at the same time as a region of medium term rates move down, or vice versa. 
In this particular example, this type of movement is only responsible for about 5.75% of the variability.

Having identified the most important factors, we can use their functional form to predict 
the most likely evolution of the yeild curve. 
Thus, a simple linear regression is fitted for the shift factor as it simply moves the curve up and down. 
Second degree polynomial is fitted for the tilt factor and higher degree can approximate flexing. 
Thus, yield curve can be approximated by linear combination of first three loadings.
"""

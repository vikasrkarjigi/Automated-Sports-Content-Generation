import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('provided_data.csv')

# Display the first 5 rows
print(df.head())

# Display basic information about the dataset
print(df.info())

# Calculate and print summary statistics
print(df.describe())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('Frame Number')
plt.ylabel('Value')
plt.title('Plot of Second Column vs Frame Number')
plt.grid(True)
plt.savefig('plot.png')
plt.show()

#Scatter Plot
plt.figure(figure=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color='blue', alpha=0.5)
plt.xlabel("Frame Number")
plt.ylabel('Value')
plt.title('Scatter Plot of Second Column vs Frame Number')
plt.grid(False)
plt.savefig('scatter_plot.png')
plt.show()

#Multiple columns plot
plt.figure(figure=(10, 6))
plt.plot(df.iloc[:, 0], df.iloc[:, 1], color='blue', alpha=0.3, label='column2')
plt.plot(df.iloc[:, 0], df.iloc[:, 2], color='red', alpha=0.4, label='column3')
plt.plot(df.iloc[:, 0], df.iloc[:, 3], color='green', alpha=0.8, label='column4')
plt.plot(df.iloc[:, 0], df.iloc[:, 4], color='yellow', alpha=0.7, label='column5')
plt.xlabel('Frame Number')
plt.ylabel('Value')
plt.title('Multiple Column vs Frame Number')
plt.legend()
plt.savefig('multiple_column_plot.png')
plt.show()

#Histogram Plot
plt.figure(figure=(10, 6))
plt.hist(df.iloc[:, 2], bins=30, color='orange')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Column 3')
plt.grid(True)
plt.savefig('Histogram_plot.png')
plt.show()

# Subplots for Multiple Visualizations
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Line plot
axs[0, 0].plot(df.iloc[:, 0], df.iloc[:, 1], color='blue')
axs[0, 0].set_title('Line Plot')
axs[0, 0].set_xlabel('Frame Number')
axs[0, 0].set_ylabel('Value')

# Scatter plot
axs[0, 1].scatter(df.iloc[:, 0], df.iloc[:, 3], color='red')
axs[0, 1].set_title('Scatter Plot')
axs[0, 1].set_xlabel('Frame Number')
axs[0, 1].set_ylabel('Value')

# Histogram
axs[1, 0].hist(df.iloc[:, 1], bins=30, color='green')
axs[1, 0].set_title('Histogram')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')

# Box plot
axs[1, 1].boxplot(df.iloc[:, 3])
axs[1, 1].set_title('Box Plot')
axs[1, 1].set_ylabel('Value')

plt.tight_layout()
plt.savefig('Subplots.png')
plt.show()

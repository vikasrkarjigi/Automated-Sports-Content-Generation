import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/provided_data.csv')

# Calculate the min and max values for the assumed X and Y coordinate columns (columns 1 and 2)
x_min, x_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()
y_min, y_max = df.iloc[:, 2].min(), df.iloc[:, 2].max()

print(f"x_min = {x_min} , x_max = {x_max} , y_min = {y_min} , y_max = {y_max}")

# Define the region boundaries
x_region_min, x_region_max = -0.5, 0.5
y_region_min, y_region_max = -0.5, 0.5

# Filter the data to identify points within the region
in_region = (df.iloc[:, 1] >= x_region_min) & (df.iloc[:, 1] <= x_region_max) & \
            (df.iloc[:, 2] >= y_region_min) & (df.iloc[:, 2] <= y_region_max)

# Calculate time spent in the region (number of frames)
time_in_region = in_region.sum()

print(f"Time spent in the specified region: {time_in_region} frames")

# Visualize data points that are in the region
plt.figure(figsize=(8,6))
plt.scatter(df.iloc[:, 1], df.iloc[:, 2], label='All Points', color='lightblue')
plt.scatter(df[in_region].iloc[:, 1], df[in_region].iloc[:, 2], label='Points in Region', color='orange')
plt.axvline(x_region_min, color='red', linestyle='--')
plt.axvline(x_region_max, color='red', linestyle='--')
plt.axhline(y_region_min, color='red', linestyle='--')
plt.axhline(y_region_max, color='red', linestyle='--')
plt.xlabel('Frame Number')
plt.ylabel('Value')
plt.title(f'Time spent in the specified region ( {x_region_min}, {x_region_max}): {time_in_region} frames')
plt.legend()
plt.savefig('data_preprocessing/results/Time_in_specific_region.png')
plt.show()

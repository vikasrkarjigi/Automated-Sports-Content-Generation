import pandas as pd
import cv2
import numpy as np

# Read the CSV file
df = pd.read_csv('data/provided_data.csv')

# Display the first 5 rows
print(df.head())

# Display basic information about the dataset
print(df.info())

# Calculate and print summary statistics
print(df.describe())

def create_animation(df):
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data_preprocessing/results/animation.mp4', fourcc, 30.0, (800, 600))

    # Normalize coordinates to fit within the frame
    x_min, x_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()
    y_min, y_max = df.iloc[:, 2].min(), df.iloc[:, 2].max()
    
    for _, row in df.iterrows():
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Normalize and scale coordinates
        x = int((row.iloc[1] - x_min) / (x_max - x_min) * 780 + 10)
        y = int((row.iloc[2] - y_min) / (y_max - y_min) * 580 + 10)
        
        # Draw the point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {int(row.iloc[0])}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()

# Create the animation
create_animation(df)

print("Animation saved as 'animation.mp4'")

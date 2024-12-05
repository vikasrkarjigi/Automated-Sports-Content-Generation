import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import ParameterSampler

# Load provided_data.csv
data = pd.read_csv('data/provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])

# Convert 'effort' column to numeric; non-numeric entries will be set to NaN
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')

# Impute missing 'effort' values using linear interpolation
data['effort'] = data['effort'].interpolate(method='linear')

# Ensure 'frame' is integer type for merging
data['frame'] = data['frame'].astype(int)

# Load target.csv
target = pd.read_csv('data/target.csv')  # Assumes columns 'frame' and 'value'

# Ensure 'frame' is integer type for merging
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Features and target
features = ['xc', 'yc', 'w', 'h', 'effort']
X = merged[features]
y = merged['value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to create lag features for time series data
def create_lag_features(X, window_size):
    X_lagged = pd.DataFrame()
    for i in range(window_size):
        X_shifted = pd.DataFrame(X).shift(i)
        X_shifted.columns = [f"{col}_lag_{i}" for col in X_shifted.columns]
        X_lagged = pd.concat([X_lagged, X_shifted], axis=1)
    return X_lagged.dropna()

window_size = 2  # Define the window size for time series chunks
X_lagged = create_lag_features(X_scaled, window_size)
y_lagged = y.iloc[window_size - 1:]  # Adjust y to align with lagged features
frames_lagged = merged['frame'].iloc[window_size - 1:]  # Get corresponding frame numbers

# Align indices
y_lagged = y_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
frames_lagged = frames_lagged.iloc[:len(X_lagged)].reset_index(drop=True)
X_lagged = X_lagged.reset_index(drop=True)

# Split into train and test sets (chronological split to respect time series nature)
split_index = int(len(X_lagged) * 0.7)
X_train = X_lagged.iloc[:split_index]
X_test = X_lagged.iloc[split_index:]
y_train = y_lagged.iloc[:split_index]
y_test = y_lagged.iloc[split_index:]
frames_test = frames_lagged.iloc[split_index:]  # Frames corresponding to test set

# Set up SVM model
svm_model = SVC(random_state=42)

# Optimized hyperparameter tuning
param_dist = {
    'C': [0.1, 1, 10],  # Reduce range
    'kernel': ['linear', 'rbf'],  # Use only commonly effective kernels
    'gamma': ['scale'],  # Fixed to 'scale' for faster performance
    'class_weight': [None]  # Simplify class weighting
}

n_iter = 5  # Reduced number of combinations
param_sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=42)

best_model = None
best_score = -1
best_params = None

# Progress bar
with tqdm(total=n_iter, desc="Hyperparameter Tuning", unit=" iteration") as pbar:
    for params in param_sampler:
        # Initialize the model with sampled parameters
        model = SVC(**params, random_state=42)
        try:
            # Train the model
            model.fit(X_train, y_train)
            # Evaluate on the validation data
            score = model.score(X_test, y_test)
            # Update the best model if current score is higher
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
        except Exception as e:
            print(f"Skipping parameters {params} due to error: {e}")
        pbar.update(1)

# Best parameters and score
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# Predict on the test set using the best model
y_pred = best_model.predict(X_test)

# Compute and print classification report
print(classification_report(y_test, y_pred))

# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': frames_test, 'value': y_pred})
predictions_df.to_csv('model_training/results/predictions_svm.csv', index=False)

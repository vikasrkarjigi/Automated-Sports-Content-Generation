import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, ParameterGrid
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load datasets

data = pd.read_csv('provided_data.csv', header=None, names=['frame', 'xc', 'yc', 'w', 'h', 'effort'])
target = pd.read_csv('target.csv')  # Ensure columns 'frame' and 'value'

# Data Cleaning and Feature Engineering
data['effort'] = pd.to_numeric(data['effort'], errors='coerce')
data['effort'] = data['effort'].interpolate(method='linear')
data['frame'] = data['frame'].astype(int)
target['frame'] = target['frame'].astype(int)

# Merge data and target on 'frame'
merged = pd.merge(data, target, on='frame', how='inner')

# Feature Engineering
merged['aspect_ratio'] = merged['w'] / merged['h']
merged['size'] = merged['w'] * merged['h']
features = ['xc', 'yc', 'w', 'h', 'effort', 'aspect_ratio', 'size']

# Prepare features and target
X = merged[features]
y = merged['value']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_scaled, y, merged.index, test_size=0.3, random_state=42
)


# SVM Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['rbf', 'poly', 'linear']  # Different kernel types
}

# Store all results (including F1 scores for each parameter combination)
results = []

# Iterate over parameter grid manually with tqdm progress bar
for params in tqdm(list(ParameterGrid(param_grid)), desc="Hyperparameter Tuning Progress"):
    # Set parameters
    svm = SVC(**params)
    svm.fit(X_train, y_train)

    # Calculate F1 scores on test set
    y_pred = svm.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # Store results for plotting
    results.append({'C': params['C'], 'gamma': params['gamma'], 'kernel': params['kernel'],
                    'macro_f1': macro_f1, 'weighted_f1': weighted_f1})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create pivot tables for macro and weighted F1 scores
pivot_table_macro = results_df.pivot_table(values='macro_f1', index='C', columns='gamma')
pivot_table_weighted = results_df.pivot_table(values='weighted_f1', index='C', columns='gamma')

# Plot the heatmap for weighted F1 scores
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table_weighted, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('SVM Hyperparameter Tuning (F1 Score)')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.tight_layout()
plt.savefig('weighted_f1_heatmap.png')
plt.show()

# Optional: Plot macro F1 score as well (for comparison)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table_macro, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('SVM Hyperparameter Tuning (Macro F1 Score)')
plt.xlabel('Gamma')
plt.ylabel('C')
plt.tight_layout()
plt.savefig('macro_f1_heatmap.png')
plt.show()

# Train the best model on the entire training data
best_params = results_df.loc[results_df['weighted_f1'].idxmax()].to_dict()
best_svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
best_svm.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_svm.predict(X_test)
print(classification_report(y_test, y_pred))
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Best parameters: {best_params}")
print(f"Final Weighted F1 Score on Test Data: {f1:.3f}")

predictions_df = pd.DataFrame({'frame': merged.loc[indices_test, 'frame'], 'value': y_pred})
predictions_df.to_csv('predictions_svm.csv', index=False)

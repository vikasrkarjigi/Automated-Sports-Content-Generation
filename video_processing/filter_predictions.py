from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class PredictionSmoother:
    def __init__(self):
        self.best_params = None
        self.best_metrics = None

    def smooth_predictions(self, raw_predictions, sigma=1.0, min_duration=10):
        # Apply Gaussian smoothing
        smoothed_gaussian = gaussian_filter1d(raw_predictions, sigma=sigma)
        smoothed_gaussian = np.round(smoothed_gaussian)  # Convert to binary prediction (0 or 1)

        # Apply Majority Voting as an ensemble technique
        smoothed_majority = self.majority_voting_smoothing(raw_predictions, window_size=5)

        # Combine Gaussian and Majority voting predictions
        smoothed_combined = np.round((smoothed_gaussian + smoothed_majority) / 2)

        # Post-process to enforce minimum duration for prediction consistency
        smoothed_combined = self.enforce_min_duration(smoothed_combined, min_duration)

        return smoothed_combined

    def enforce_min_duration(self, predictions, min_duration):
        """Ensure that predictions are consistent for at least `min_duration` frames"""
        smoothed_predictions = predictions.copy()

        # Check if a prediction persists for at least min_duration frames
        for i in range(len(predictions) - min_duration + 1):
            if np.all(predictions[i:i + min_duration] == 1):
                smoothed_predictions[i:i + min_duration] = 1
            elif np.all(predictions[i:i + min_duration] == 0):
                smoothed_predictions[i:i + min_duration] = 0

        return smoothed_predictions

    def majority_voting_smoothing(self, raw_predictions, window_size=5):
        """Applies majority voting within a sliding window"""
        smoothed_predictions = []
        for i in range(len(raw_predictions)):
            start = max(0, i - window_size // 2)
            end = min(len(raw_predictions), i + window_size // 2 + 1)
            window = raw_predictions[start:end]
            smoothed_predictions.append(int(np.mean(window) > 0.5))  # Majority voting
        return np.array(smoothed_predictions)

    def optimize_parameters(self, raw_predictions, target_sequence, metric='f1_score'):
        param_grid = {
            'sigma': np.linspace(0.1, 5.0, 50),  # Extended range for sigma
            'min_duration': [5, 7, 10, 12, 15, 20],  # Extended values for min_duration
        }

        best_score = -np.inf
        best_params = None
        best_metrics = None

        for sigma in param_grid['sigma']:
            for min_duration in param_grid['min_duration']:
                smoothed_predictions = self.smooth_predictions(raw_predictions, sigma=sigma, min_duration=min_duration)
                metrics = self.evaluate_metrics(smoothed_predictions, target_sequence)

                if metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_params = {'sigma': sigma, 'min_duration': min_duration}
                    best_metrics = metrics

        self.best_params = best_params
        self.best_metrics = best_metrics

        return best_params, best_metrics

    def evaluate_metrics(self, smoothed_predictions, target_sequence):
        accuracy = accuracy_score(target_sequence, smoothed_predictions)
        f1 = f1_score(target_sequence, smoothed_predictions)
        precision = precision_score(target_sequence, smoothed_predictions)
        recall = recall_score(target_sequence, smoothed_predictions)

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    def save_params(self, params, file_name):
        with open(file_name, 'w') as f:
            json.dump(params, f)

    def load_params(self, file_name):
        with open(file_name, 'r') as f:
            return json.load(f)


def plot_predictions(raw_predictions, smoothed_predictions, target_sequence, output_file):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(target_sequence, label='Target', alpha=0.7)
    plt.plot(raw_predictions, label='Raw Predictions', alpha=0.5)
    plt.plot(smoothed_predictions, label='Smoothed Predictions', alpha=0.7)

    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.legend()
    plt.title('Predictions Comparison')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    # Read CSV files
    predictions_df = pd.read_csv('model_training/results/predictions_svm_hyperparameter.csv')
    target_df = pd.read_csv('data/target.csv')

    # Merge dataframes on 'frame' column and sort by frame
    merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')

    # Convert values to integers (0 and 1)
    raw_predictions = merged_df['value_pred'].astype(int).tolist()
    target_sequence = merged_df['value_target'].astype(int).tolist()

    # Create smoother and optimize parameters
    smoother = PredictionSmoother()
    best_params, best_metrics = smoother.optimize_parameters(
        raw_predictions, target_sequence, metric='f1_score'
    )

    # Apply smoothing with best parameters
    smoothed_predictions = smoother.smooth_predictions(raw_predictions, best_params['sigma'],
                                                       best_params['min_duration'])

    # Print results
    print("\nBest parameters found:")
    print(json.dumps(best_params, indent=2))
    print("\nMetrics with best parameters:")
    print(json.dumps(best_metrics, indent=2))

    # Create a new dataframe with smoothed predictions
    result_df = pd.DataFrame({
        'frame': merged_df['frame'],
        'value': smoothed_predictions
    })

    # Write the result to a new CSV file
    result_df.to_csv('video_processing/results/smoothed_predictions.csv', index=False)
    print("\nSmoothed predictions have been written to 'smoothed_predictions.csv'")

    # Create and save the plot
    plot_predictions(raw_predictions, smoothed_predictions, target_sequence, 'video_processing/results/predictions_comparison.png')
    print("\nPredictions comparison plot has been saved to 'predictions_comparison.png'")

    # Example of saving and loading parameters
    smoother.save_params(best_params, "video_processing/results/best_smoothing_params.json")
    loaded_params = smoother.load_params("video_processing/results/best_smoothing_params.json")
    print("\nLoaded parameters from 'best_smoothing_params.json':")
    print(json.dumps(loaded_params, indent=2))

import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('wandb_export_2024-11-30T16_14_22.441-05_00.csv')

# Get the column names for the main metrics (excluding MIN/MAX columns)
metric_columns = [col for col in df.columns if not (col.endswith('__MIN') or col.endswith('__MAX') or col == 'Step')]

# Calculate rolling average with window size 10 for each metric
window_size = 10
rolling_averages = {}
best_windows = {}

for column in metric_columns:
    # Calculate rolling mean
    rolling_avg = df[column].rolling(window=window_size).mean()
    rolling_averages[column] = rolling_avg
    
    # Find the best window and its location (ignoring NaN values)
    rolling_avg_clean = rolling_avg.dropna()
    if not rolling_avg_clean.empty:
        best_value = rolling_avg_clean.max()
        best_index = rolling_avg_clean.idxmax()
        start_step = df['Step'][max(0, best_index - window_size + 1)]
        end_step = df['Step'][best_index]
        
        best_windows[column] = {
            'average': best_value,
            'start_step': start_step,
            'end_step': end_step
        }

# Print results
for column in metric_columns:
    if column in best_windows:
        result = best_windows[column]
        print(f"\nMetric: {column}")
        print(f"Best {window_size}-point window average: {result['average']:.2f}")
        print(f"Location: Steps {result['start_step']} to {result['end_step']}")
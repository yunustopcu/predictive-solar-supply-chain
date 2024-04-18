import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ratio_change_accuracy(y_true, y_pred):
    # Convert to numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Ensure no zero values to avoid division by zero
    y_true += 1e-9
    y_pred += 1e-9
    
    # Calculate ratios for actual and predicted
    ratios_true = y_true[1:] / y_true[:-1]
    ratios_pred = y_pred[1:] / y_pred[:-1]
    
    # Calculate the absolute differences in ratios
    ratio_diffs = np.abs(ratios_true - ratios_pred)
    
    # Return the average of these differences
    return 100 * (np.mean(ratio_diffs))




# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Load the CSV files
ets_df = pd.read_csv('/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/ets_predictions_and_actuals_2017.csv')
lstm_df = pd.read_csv('/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/lstm_predictions_and_actuals_2017.csv')

# Clean up and merge the dataframes
lstm_df.drop(columns=['Unnamed: 0'], inplace=True)
lstm_df.rename(columns={'Prediction': 'LSTM_Prediction'}, inplace=True)
ets_df.rename(columns={'Prediction': 'ETS_Prediction'}, inplace=True)
merged_df = pd.merge(ets_df, lstm_df, on='Date')

# Convert 'Date' column to datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Calculate the average of ETS and LSTM predictions
merged_df['Average_Prediction'] = (merged_df['ETS_Prediction'] + merged_df['LSTM_Prediction']) / 2

# Extract month and year for filtering
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Year'] = merged_df['Date'].dt.year

# Filter for 2017 and months February to December
filtered_df = merged_df[(merged_df['Year'] == 2017) & (merged_df['Month'] >= 2) & (merged_df['Month'] <= 12)]

# Group data by month and plot
for month, group in filtered_df.groupby('Month'):
    plt.figure(figsize=(15, 5))
    plt.plot(group['Date'], group['Actual'], label='Actual Data', color='blue')
    plt.plot(group['Date'], group['ETS_Prediction'], label='ETS Prediction', color='green')
    plt.plot(group['Date'], group['LSTM_Prediction'], label='LSTM Prediction', color='red')
    plt.plot(group['Date'], group['Average_Prediction'], label='Average Prediction', color='purple', linestyle='--')
    plt.title(f'Actual vs. Predictions for Month {month} of 2017')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate and print the average hourly MAPE for ETS, LSTM, and their average
    mape_ets = mean_absolute_percentage_error(group['Actual'], group['ETS_Prediction'])
    mape_lstm = mean_absolute_percentage_error(group['Actual'], group['LSTM_Prediction'])
    mape_avg = mean_absolute_percentage_error(group['Actual'], group['Average_Prediction'])
    print(f"Month {month}: Average hourly MAPE for ETS: {mape_ets:.2f}%")
    print(f"Month {month}: Average hourly MAPE for LSTM: {mape_lstm:.2f}%")
    print(f"Month {month}: Average hourly MAPE for Average Prediction: {mape_avg:.2f}%\n")

    print("---------------")
    
    # Calculate the custom ratio change accuracy for ETS, LSTM, and their average
    ratio_accuracy_ets = ratio_change_accuracy(group['Actual'], group['ETS_Prediction'])
    ratio_accuracy_lstm = ratio_change_accuracy(group['Actual'], group['LSTM_Prediction'])
    ratio_accuracy_avg = ratio_change_accuracy(group['Actual'], group['Average_Prediction'])

    # Print the custom ratio change accuracy for each
    print(f"Month {month}: Custom Ratio Change Accuracy for ETS: {ratio_accuracy_ets:.4f}")
    print(f"Month {month}: Custom Ratio Change Accuracy for LSTM: {ratio_accuracy_lstm:.4f}")
    print(f"Month {month}: Custom Ratio Change Accuracy for Average Prediction: {ratio_accuracy_avg:.4f}\n")



# Select only the columns you want to save to the CSV
columns_to_save = ['Date', 'Average_Prediction']
final_df_to_save = merged_df[columns_to_save]

# Save the DataFrame to a CSV file
final_df_to_save.to_csv('averaged_predictions_2017.csv', index=False)

print("CSV file has been saved.")

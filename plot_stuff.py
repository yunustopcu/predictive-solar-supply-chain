import pandas as pd
import matplotlib.pyplot as plt

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
    plt.title(f'Actual vs. Predictions for Month {month} of 2017')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions and actual values from the CSV file
csv_path = '/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/predictions_and_actuals_2017.csv'
data = pd.read_csv(csv_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plotting by month
for month in range(1, 13):
    plt.figure(figsize=(10, 5))
    monthly_data = data[data.index.month == month]
    
    plt.plot(monthly_data.index, monthly_data['Actual'], label='Actual Values', alpha=0.7)
    plt.plot(monthly_data.index, monthly_data['Prediction'], label='Predicted Values', alpha=0.7, linestyle='--')
    plt.title(f'Actual vs Predicted Values for {monthly_data.index.month_name()[0]} 2017')
    plt.xlabel('Date')
    plt.ylabel('Value (kWh)')
    plt.legend()
    plt.tight_layout()
    
    # Adjusting the x-axis to show more readable date labels
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of x-axis labels shown
    
    plt.show()

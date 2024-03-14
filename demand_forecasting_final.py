import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from datetime import datetime, timedelta


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing


file_path = '/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/power_usage_2016_to_2020.csv'
data = pd.read_csv(file_path)


from datetime import datetime

# Define a function to check if the date is in the expected format
def check_date_format(date_string):
    try:
        # If the date can be parsed in the expected format, it's consistent
        datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        # If parsing fails, the format is inconsistent
        return False

# Apply the check to the 'StartDate' column
date_format_consistency = data['StartDate'].apply(check_date_format)

# Check if there are any False values in the results, indicating inconsistencies
inconsistencies = not date_format_consistency.all()

# Count the number of inconsistent dates, if any
inconsistent_dates_count = len(data[~date_format_consistency])

inconsistencies, inconsistent_dates_count


# To find the first and last date in the dataset, we'll convert the 'StartDate' column to datetime format and then find the minimum and maximum dates.

# Convert 'StartDate' to datetime format
#data['StartDate'] = pd.to_datetime(data['StartDate'])

n = data.shape[0]
p1 = pd.Series(range(n), pd.period_range('2016-06-01 00:00:00', freq = '1H', periods = n))
data['StartDate'] = p1.to_frame().index


# Find the first (minimum) and last (maximum) date
first_date = data['StartDate'].min()
last_date = data['StartDate'].max()

first_date, last_date

# DEBUGGING LINE, DONT BOTHER
#data.to_csv('data.csv', index=False)


# To trim the dataset to only include data from 2017-01-01 to 2018-01-01, we'll filter the 'StartDate' to be within this range.

start_date = '2017-01-01'
end_date = '2018-01-01'

# Filter the dataset for the given date range
trimmed_data = data[(data['StartDate'] >= start_date) & (data['StartDate'] <= end_date)]

# Check the first few rows to ensure the trimming was successful
trimmed_data.head(), trimmed_data.tail()


# Convert 'StartDate' from PeriodDtype to datetime
trimmed_data['StartDate'] = trimmed_data['StartDate'].dt.to_timestamp()
trimmed_data['StartDate'] = pd.to_datetime(trimmed_data['StartDate'])
trimmed_data.set_index('StartDate', inplace=True)
trimmed_data = trimmed_data.asfreq('H', method='ffill')  # Ensure hourly frequency, filling any missing values



# Function to fit and predict using ExponentialSmoothing
def fit_predict_ets(training_data, prediction_length):
    model = ExponentialSmoothing(training_data, seasonal='add', seasonal_periods=24).fit()
    forecast = model.forecast(prediction_length)
    return forecast

# Initialize an empty DataFrame to store predictions and actual values
predictions = pd.DataFrame()

# Define the time frame for prediction (iterating through 2017)
start_date = pd.Timestamp('2017-01-31')
end_date = pd.Timestamp('2017-12-31')

# Iterating through each day of 2017 for prediction after the first 30 days
while start_date <= end_date:
    training_start = start_date - timedelta(days=30)
    training_end = start_date
    
    # Extract the training data
    training_data = trimmed_data.loc[training_start:training_end]['Value (kWh)']
    
    # Predict the next 24 hours
    forecast = fit_predict_ets(training_data, 24)
    
    # Prepare forecast DataFrame
    forecast_dates = pd.date_range(start=start_date, periods=24, freq='H')
    temp_df = pd.DataFrame({'Date': forecast_dates, 'Prediction': forecast.values, 'Actual': trimmed_data.loc[forecast_dates, 'Value (kWh)']})
    
    # Append the predictions
    predictions = pd.concat([predictions, temp_df])
    
    # Move to the next day
    start_date += timedelta(days=1)

# Calculate MAPE
predictions.dropna(inplace=True)  # Ensure there are no NaN values
mape = (np.abs(predictions['Actual'] - predictions['Prediction']) / predictions['Actual']).mean() * 100

# Reset index of the predictions DataFrame and save it to a CSV file
predictions.reset_index(drop=True, inplace=True)
predictions.to_csv('predictions_and_actuals_2017.csv', index=False)

print(f"Predictions and actual values for 2017 have been saved successfully. MAPE for the entire year: {mape:.2f}%.")


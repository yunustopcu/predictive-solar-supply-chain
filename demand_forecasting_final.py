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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam




file_path = '/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/power_usage_2016_to_2020.csv'
data = pd.read_csv(file_path)


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


original_trimmed_data = trimmed_data

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
predictions.to_csv('ets_predictions_and_actuals_2017.csv', index=False)

print(f"Predictions and actual values for ETS 2017 have been saved successfully. MAPE for the entire year: {mape:.2f}%.")


trimmed_data = original_trimmed_data



def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)



def lstm_prediction_moving_window(trimmed_data, start_date, prediction_length=24):
    """
    Function to make predictions using an LSTM model based on a moving 30-day window.
    Predicts the next 24 hours by updating the model input with the latest prediction.

    Parameters:
    - trimmed_data: DataFrame containing the trimmed and preprocessed data.
    - start_date: The starting date for prediction.
    - prediction_length: Number of hours to predict. Default is 24.

    Returns:
    - A DataFrame containing the predictions and the corresponding dates.
    """
    # Ensure the index is in datetime format
    if not isinstance(trimmed_data.index, pd.DatetimeIndex):
        trimmed_data.index = pd.to_datetime(trimmed_data.index)

    training_start = start_date - pd.Timedelta(days=30)
    training_end = start_date

    # Select the training data
    train_data = trimmed_data.loc[training_start:training_end]['Value (kWh)'].values

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

    # Create the dataset for LSTM
    look_back = 24  # considering 24 hours pattern
    X_train, Y_train = create_dataset(train_data_scaled, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # LSTM Model configuration
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Fit the model
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

    # Initialize the list to store predictions
    predictions_rescaled_list = []

    # Start with the last look_back hours as the initial input
    current_input = np.array(train_data_scaled[-look_back:]).reshape(1, look_back, 1)

    for _ in range(prediction_length):
        # Predict the next step
        current_prediction = model.predict(current_input)
        # Rescale the prediction to the original data scale
        current_prediction_rescaled = scaler.inverse_transform(current_prediction)
        # Store the rescaled prediction
        predictions_rescaled_list.append(current_prediction_rescaled[0][0])

        # Update the current_input to include the latest prediction and exclude the oldest data point
        current_input = np.append(current_input[0][1:], current_prediction).reshape(1, look_back, 1)
    
    # Prepare the DataFrame with predictions
    forecast_dates = pd.date_range(start=start_date, periods=prediction_length, freq='H')
    predictions_df = pd.DataFrame({'Date': forecast_dates, 'Prediction': predictions_rescaled_list})

    return predictions_df

# Initialize an empty DataFrame to store LSTM predictions
lstm_predictions = pd.DataFrame()

# Assuming `trimmed_data` is already filtered to the desired date range and set with 'StartDate' as the index
start_date = pd.Timestamp('2017-01-31')
end_date = pd.Timestamp('2017-12-31')

while start_date <= end_date:
    lstm_preds = lstm_prediction_moving_window(trimmed_data, start_date)
    lstm_predictions = pd.concat([lstm_predictions, lstm_preds])
    
    # Increment the start_date by one day for the next iteration
    start_date += pd.Timedelta(days=1)# You can save the LSTM predictions to a CSV file or further analyze them as needed


    
lstm_predictions.to_csv('lstm_predictions_and_actuals_2017.csv')

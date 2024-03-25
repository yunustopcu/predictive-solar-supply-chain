import pandas as pd

file_path = '/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/averaged_predictions_2017.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date']) 
df['Hour'] = df['Date'].dt.hour

# Calculate the average energy consumption for each hour across all days
average_hourly_consumption = df[df['Average_Prediction'] >= 0].groupby('Hour')['Average_Prediction'].mean()

# Replace negative energy values with the average for their respective hour
def replace_negative_with_hourly_average(row):
    if row['Average_Prediction'] < 0:
        return average_hourly_consumption[row['Hour']]
    else:
        return row['Average_Prediction']

df['Average_Prediction'] = df.apply(replace_negative_with_hourly_average, axis=1)

# Save the modified dataset
output_file_path = '/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/non_negative_averaged_predictions_2017.csv'
df.drop(columns=['Hour'], inplace=True)
df.to_csv(output_file_path, index=False)

print("Negative energy values have been replaced and the new dataset is saved.")

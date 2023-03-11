import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# load data for high-frequency product
hf_data = pd.read_csv('clean_sales_fabric_96_sku_265_warehouse_1.csv', parse_dates=['date'])
hf_data = hf_data.groupby('date').sum()

# load data for low-frequency product
lf_data = pd.read_csv('clean_sales_fabric_42_sku_13653_warehouse_-1.csv', parse_dates=['date'])
lf_data = lf_data.groupby('date').sum()

# resample data to daily frequency and fill in missing values using interpolation
hf_data = hf_data.resample('D').sum().interpolate(method='linear')
lf_data = lf_data.resample('D').sum().interpolate(method='linear')

# plot the high-frequency and low-frequency data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(hf_data.index, hf_data['count'])
ax1.set(title='High-frequency product sales', xlabel='Date', ylabel='Sales')
ax2.plot(lf_data.index, lf_data['count'])
ax2.set(title='Low-frequency product sales', xlabel='Date', ylabel='Sales')
plt.show()

# fit ARIMA models to the high-frequency and low-frequency data
hf_model = ARIMA(hf_data['count'], order=(1, 1, 0)).fit()
lf_model = ARIMA(lf_data['count'], order=(1, 1, 0)).fit()

# generate forecasts for the next 30-40 days using the fitted models
hf_forecast = hf_model.forecast(steps=40)
lf_forecast = lf_model.forecast(steps=30)

# plot the high-frequency and low-frequency forecasts
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(hf_data.index, hf_data['count'], label='Actual')
ax1.plot(hf_forecast.index, hf_forecast, label='Forecast')
ax1.set(title='High-frequency product sales forecast', xlabel='Date', ylabel='Sales')
ax1.legend()
ax2.plot(lf_data.index, lf_data['count'], label='Actual')
ax2.plot(lf_forecast.index, lf_forecast, label='Forecast')
ax2.set(title='Low-frequency product sales forecast', xlabel='Date', ylabel='Sales')
ax2.legend()
plt.show()

# calculate buying score for the high-frequency and low-frequency forecasts
hf_actual = hf_data['count'].iloc[-1]
hf_predicted = hf_forecast[-1]
hf_score = (1 + hf_predicted) / (1 + hf_actual)

lf_actual = lf_data['count'].iloc[-1]
lf_predicted = lf_forecast[-1]
lf_score = (1 + lf_predicted) / (1 + lf_actual)

print("High-frequency product buying score:", hf_score)
print("Low-frequency product buying score:", lf_score)

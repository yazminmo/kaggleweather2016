#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1.load dataset- confirm there is an active working directory
import pandas as pd

import os
working_directory = os.getcwd()
print(working_directory)


# In[2]:


# 1.1 load dataset- import the csv file and show headers 
path = working_directory + '/Documents/Forecasting/data_prueba_Forecasting.csv'
df_bonus = pd.read_csv(path)
df_bonus.head()


# In[3]:


#defining unique as a group within chunkID
def unique(chunkID):    
    chunk_ids = unique(values[:, 1])


# In[4]:


# sort rows by chunk id
chunks = dict()
def values(chunk_id):
    for chunk_id in chunk_ids:
        selection = values[:, chunk_ix] == chunk_id
        chunks[chunk_id] = values[selection, :]


# In[5]:


# split the dataset by 'chunkID', and return a dict of id to rows
def to_chunks(values, chunk_ix=1):
        chunks = dict()
        
        # get the unique chunk ids
        chunk_ids = unique(values[:, chunk_ix])
        
        # group rows by chunk id
        for chunk_id in chunk_ids:
                selection = values[:, chunk_ix] == chunk_id
                chunks[chunk_id] = values[selection, :]
        return chunks


# In[12]:


# load data and split into chunks
from numpy import unique
from pandas import read_csv
 


# In[13]:


# split the dataset by 'chunkID', return a dict of id to rows
def to_chunks(values, chunk_ix=1):
        chunks = dict()
        # get the unique chunk ids
        chunk_ids = unique(values[:, chunk_ix])
        # group rows by chunk id
        for chunk_id in chunk_ids:
                selection = values[:, chunk_ix] == chunk_id
                chunks[chunk_id] = values[selection, :]
        return chunks


# In[15]:


# load dataset
dataset =read_csv('/Users/yazmintorresmontana/Documents/Forecasting/data_prueba_Forecasting.csv', header=0)  



# In[8]:


# group data by chunks and print the total
values = dataset.values
chunks = to_chunks(values)
print('Total Chunks: %d' % len(chunks))


# In[16]:


# split each chunk into train/test sets
def split_train_test(chunks, row_in_chunk_ix=2):
        train, test = list(), list()
        # first 5 days of hourly observations for train
        cut_point = 5 * 24
        # enumerate chunks
        for k,rows in chunks.items():
                # split chunk rows by 'position_within_chunk'
                train_rows = rows[rows[:,row_in_chunk_ix] <= cut_point, :]
                test_rows = rows[rows[:,row_in_chunk_ix] > cut_point, :]
                if len(train_rows) == 0 or len(test_rows) == 0:
                        print('>dropping chunk=%d: train=%s, test=%s' % (k, train_rows.shape, test_rows.shape))
                        continue
                # store with chunk id, position in chunk, hour and all targets
                indices = [1,2,5] + [x for x in range(56,train_rows.shape[1])]
                train.append(train_rows[:, indices])
                test.append(test_rows[:, indices])
        return train, test
    


# In[17]:



# return a list of relative forecast lead times
def get_lead_times():
        return [1, 2 ,3, 4, 5, 10, 17, 24, 48, 72]
    


# In[18]:


# convert the rows in a test chunk to forecasts
def to_forecasts(test_chunks, row_in_chunk_ix=1):
        # get lead times
        lead_times = get_lead_times()
        # first 5 days of hourly observations for train
        cut_point = 5 * 24
        forecasts = list()
        # enumerate each chunk
        for rows in test_chunks:
                chunk_id = rows[0, 0]
                # enumerate each lead time
                for tau in lead_times:
                        # determine the row in chunk we want for the lead time
                        offset = cut_point + tau
                        # retrieve data for the lead time using row number in chunk
                        row_for_tau = rows[rows[:,row_in_chunk_ix]==offset, :]
                        # check if we have data
                        if len(row_for_tau) == 0:
                                # create a mock row [chunk, position, hour] + [nan...]
                            row = [chunk_id, offset, nan] + [nan for _ in range(39)]
                            forecasts.append(row)
                        else:
                                # store the forecast row
                                forecasts.append(row_for_tau[0])
        return array(forecasts)
    


# In[20]:


# load dataset
dataset =read_csv('/Users/yazmintorresmontana/Documents/Forecasting/data_prueba_Forecasting.csv', header=0)  
# group data by chunks
values = dataset.values
chunks = to_chunks(values)
# split into train/test
train, test = split_train_test(chunks)
# flatten training chunks to rows
train_rows = array([row for rows in train for row in rows])
# print(train_rows.shape)
print('Train Rows: %s' % str(train_rows.shape))
# reduce train to forecast lead times only
test_rows = to_forecasts(test)
print('Test Rows: %s' % str(test_rows.shape))
# save datasets
savetxt('/Users/yazmintorresmontana/Documents/Forecasting/data_prueba_Forecasting.csv', train_rows, delimiter=',')
savetxt('/Users/yazmintorresmontana/Documents/Forecasting/data_prueba_Forecasting.csv', test_rows, delimiter=',')


# In[21]:


# convert the test dataset in chunks to [chunk][variable][time] format
def prepare_test_forecasts(test_chunks):
        predictions = list()
        # enumerate chunks to forecast
        for rows in test_chunks:
                # enumerate targets for chunk
                chunk_predictions = list()
                for j in range(3, rows.shape[1]):
                        yhat = rows[:, j]
                        chunk_predictions.append(yhat)
                chunk_predictions = array(chunk_predictions)
                predictions.append(chunk_predictions)
        return array(predictions)
    


# In[22]:


# calculate the error between an actual and predicted value
def calculate_error(actual, predicted):
        # give the full actual value if predicted is nan
        if isnan(predicted):
                return abs(actual)
        # calculate abs difference
        return abs(actual - predicted)
    


# In[23]:


# evaluate a forecast in the format [chunk][variable][time]
def evaluate_forecasts(predictions, testset):
        lead_times = get_lead_times()
        total_mae, times_mae = 0.0, [0.0 for _ in range(len(lead_times))]
        total_c, times_c = 0, [0 for _ in range(len(lead_times))]
        # enumerate test chunks
        for i in range(len(test_chunks)):
                # convert to forecasts
                actual = testset[i]
                predicted = predictions[i]
                # enumerate target variables
                for j in range(predicted.shape[0]):
                        # enumerate lead times
                        for k in range(len(lead_times)):
                            # skip if actual in nan
                                if isnan(actual[j, k]):
                                    continue
                            # calculate error
                                error = calculate_error(actual[j, k], predicted[j, k])
                            # update statistics
                                total_mae += error
                                times_mae[k] += error
                                total_c += 1
                                times_c[k] += 1
        # normalize summed absolute errors
        total_mae /= total_c
        times_mae = [times_mae[i]/times_c[i] for i in range(len(times_mae))]
        return total_mae, times_mae
    


# In[24]:


# summarize scores
def summarize_error(name, total_mae, times_mae):
        # print summary
        lead_times = get_lead_times()
        formatted = ['+%d %.3f' % (lead_times[i], times_mae[i]) for i in range(len(lead_times))]
        s_scores = ', '.join(formatted)
        print('%s: [%.3f MAE] %s' % (name, total_mae, s_scores))
        # plot summary
        pyplot.plot([str(x) for x in lead_times], times_mae, marker='.')
        pyplot.show()
        


# In[25]:


# layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=5*24):
        # lay out whole series
        data = [nan for _ in range(n_steps)]
        # mark all available data
        for i in range(len(chunk_train)):
                # get position in chunk
                position = int(chunk_train[i, 1] - 1)
                # store data
                data[position] = chunk_train[i, col_ix]
        return data


# In[26]:


# plot variables horizontally with gaps for missing data
def plot_variables(chunk_train, n_vars=39):
        pyplot.figure()
        for i in range(n_vars):
                # convert target number into column number
                col_ix = 3 + i
                # mark missing obs for variable
                series = variable_to_series(chunk_train, col_ix)
                # plot
                ax = pyplot.subplot(n_vars, 1, i+1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                pyplot.plot(series)
        # show plot
        pyplot.show()


# In[27]:


# plot missing
from numpy import loadtxt
from numpy import nan
from numpy import unique
from matplotlib import pyplot
 
# split the dataset by 'chunkID', return a list of chunks
def to_chunks(values, chunk_ix=0):
        chunks = list()
        # get the unique chunk ids
        chunk_ids = unique(values[:, chunk_ix])
        # group rows by chunk id
        for chunk_id in chunk_ids:
                selection = values[:, chunk_ix] == chunk_id
                chunks.append(values[selection, :])
        return chunks
 
    # layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=5*24):
        # lay out whole series
        data = [nan for _ in range(n_steps)]
        # mark all available data
        for i in range(len(chunk_train)):
                # get position in chunk
                position = int(chunk_train[i, 1] - 1)
                # store data
                data[position] = chunk_train[i, col_ix]
        return data
 
    # plot variables horizontally with gaps for missing data
def plot_variables(chunk_train, n_vars=39):
        pyplot.figure()
        for i in range(n_vars):
                # convert target number into column number
                col_ix = 3 + i
                # mark missing obs for variable
                series = variable_to_series(chunk_train, col_ix)
                # plot
                ax = pyplot.subplot(n_vars, 1, i+1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                pyplot.plot(series)
        # show plot
        pyplot.show()
 
    # load dataset
train = loadtxt('/Users/yazmintorresmontana/Documents/Forecasting/data_prueba_Forecasting.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks(train)
# pick one chunk
rows = train_chunks[0]
# plot variables
plot_variables(rows)


# In[28]:


# interpolate series of hours (in place) in 24 hour time
def interpolate_hours(hours):
        # find the first hour
        ix = -1
        for i in range(len(hours)):
                if not isnan(hours[i]):
                        ix = i
                        break
        # fill-forward
        hour = hours[ix]
        for i in range(ix+1, len(hours)):
                # increment hour
                hour += 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
        # fill-backward
        hour = hours[ix]
        for i in range(ix-1, -1, -1):
                # decrement hour
                hour -= 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
                        


# In[96]:


# interpolate hours
from numpy import nan
from numpy import isnan
 
# interpolate series of hours (in place) in 24 hour time
def interpolate_hours(hours):
        # find the first hour
        ix = -1
        for i in range(len(hours)):
                if not isnan(hours[i]):
                        ix = i
                        break
        # fill-forward
        hour = hours[ix]
        for i in range(ix+1, len(hours)):
                # increment hour
                hour += 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
        # fill-backward
        hour = hours[ix]
        for i in range(ix-1, -1, -1):
                # decrement hour
                hour -= 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
 
    # define hours with missing data
data = [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0, nan, 2, nan, nan, nan, nan, nan, nan, 9, 10, 11, 12, 13, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
print(data)
# fill in missing hours
interpolate_hours(data)
print(data)


# In[29]:


# prepare sequence of hours for the chunk
hours = variable_to_series(rows, 2)
# interpolate hours
interpolate_hours(hours)


# In[30]:



# impute missing data
def impute_missing(rows, hours, series, col_ix):
        # count missing observations
        n_missing = count_nonzero(isnan(series))
        # calculate ratio of missing
        ratio = n_missing / float(len(series)) * 100
        # check for no data
        if ratio == 100.0:
                return series
        # impute missing using the median value for hour in the series
        imputed = list()
        for i in range(len(series)):
                if isnan(series[i]):
                        # get all rows with the same hour
                        matches = rows[rows[:,2]==hours[i]]
                        # fill with median value
                        value = nanmedian(matches[:, col_ix])
                        imputed.append(value)
                else:
                        imputed.append(series[i])
        return imputed


# In[31]:


# plot variables horizontally with gaps for missing data
def plot_variables(chunk_train, hours, n_vars=39):
        pyplot.figure()
        for i in range(n_vars):
                # convert target number into column number
                col_ix = 3 + i
                # mark missing obs for variable
                series = variable_to_series(chunk_train, col_ix)
                ax = pyplot.subplot(n_vars, 1, i+1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # imputed
                imputed = impute_missing(chunk_train, hours, series, col_ix)
                # plot imputed
                pyplot.plot(imputed)
                # plot with missing
                pyplot.plot(series)
        # show plot
        pyplot.show()


# In[101]:


# impute missing
from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import nanmedian
from matplotlib import pyplot
 
# split the dataset by 'chunkID', return a list of chunks
def to_chunks(values, chunk_ix=0):
        chunks = list()
        # get the unique chunk ids
        chunk_ids = unique(values[:, chunk_ix])
        # group rows by chunk id
        for chunk_id in chunk_ids:
                selection = values[:, chunk_ix] == chunk_id
                chunks.append(values[selection, :])
        return chunks
 
    # impute missing data
def impute_missing(rows, hours, series, col_ix):
        # count missing observations
        n_missing = count_nonzero(isnan(series))
        # calculate ratio of missing
        ratio = n_missing / float(len(series)) * 100
        # check for no data
        if ratio == 100.0:
                return series
        # impute missing using the median value for hour in the series
        imputed = list()
        for i in range(len(series)):
                if isnan(series[i]):
                        # get all rows with the same hour
                        matches = rows[rows[:,2]==hours[i]]
                        # fill with median value
                        value = nanmedian(matches[:, col_ix])
                        imputed.append(value)
                else:
                        imputed.append(series[i])
        return imputed
 
    # interpolate series of hours (in place) in 24 hour time
def interpolate_hours(hours):
        # find the first hour
        ix = -1
        for i in range(len(hours)):
                if not isnan(hours[i]):
                        ix = i
                        break
        # fill-forward
        hour = hours[ix]
        for i in range(ix+1, len(hours)):
                # increment hour
                hour += 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
        # fill-backward
        hour = hours[ix]
        for i in range(ix-1, -1, -1):
                # decrement hour
                hour -= 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
 
    # layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=5*24):
        # lay out whole series
        data = [nan for _ in range(n_steps)]
        # mark all available data
        for i in range(len(chunk_train)):
                # get position in chunk
                position = int(chunk_train[i, 1] - 1)
                # store data
                data[position] = chunk_train[i, col_ix]
        return data
 
    # plot variables horizontally with gaps for missing data
def plot_variables(chunk_train, hours, n_vars=39):
        pyplot.figure()
        for i in range(n_vars):
                # convert target number into column number
                col_ix = 3 + i
                # mark missing obs for variable
                series = variable_to_series(chunk_train, col_ix)
                ax = pyplot.subplot(n_vars, 1, i+1)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # imputed
                imputed = impute_missing(chunk_train, hours, series, col_ix)
                # plot imputed
                pyplot.plot(imputed)
                # plot with missing
                pyplot.plot(series)
        # show plot
        pyplot.show()
     
    # load dataset
train = loadtxt('/Users/yazmintorresmontana/Documents/Time series/AirQualityPrediction/naive_train.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks(train)
# pick one chunk
rows = train_chunks[0]
# prepare sequence of hours for the chunk
hours = variable_to_series(rows, 2)
# interpolate hours
interpolate_hours(hours)
# plot variables
plot_variables(rows, hours)


# In[126]:


# forecast for each chunk, returns [chunk][variable][time]
def forecast_chunks(train_chunks, test_input):
        lead_times = get_lead_times()
        predictions = list()
        # enumerate chunks to forecast
        for i in range(len(train_chunks)):
                # prepare sequence of hours for the chunk
                hours = variable_to_series(train_chunks[i], 2)
                # interpolate hours
                interpolate_hours(hours)
                # enumerate targets for chunk
                chunk_predictions = list()
                for j in range(39):
                        yhat = forecast_variable(hours, train_chunks[i], test_input[i], lead_times, j)
                        chunk_predictions.append(yhat)
                chunk_predictions = array(chunk_predictions)
                predictions.append(chunk_predictions)
        return array(predictions)


# In[127]:


# forecast all lead times for one variable
def forecast_variable(hours, chunk_train, chunk_test, lead_times, target_ix):
        # convert target number into column number
        col_ix = 3 + target_ix
        # check for no data
        if not has_data(chunk_train[:, col_ix]):
                forecast = [nan for _ in range(len(lead_times))]
                return forecast
        # get series
        series = variable_to_series(chunk_train, col_ix)
        # impute
        imputed = impute_missing(chunk_train, hours, series, col_ix)
        # fit AR model and forecast
        forecast = fit_and_forecast(imputed)
        return forecast


# In[136]:


# define the model
def ARIMA (series, order=(1,0,0)):
    
        model = ARIMA(series, order=(1,0,0))
        


# In[145]:


# fit AR model and generate a forecast
def fit_and_forecast(series):
        # define the model
        model = ARIMA(series, order=(1,0,0))
        # return a nan forecast in case of exception
        try:
                # ignore statsmodels warnings
                with catch_warnings():
                        filterwarnings("ignore")
                        # fit the model
                        model_fit = model.fit()
                    # forecast 72 hours
                        yhat = model_fit.predict(len(series), len(series)+72)
                        # extract lead times
                        lead_times = array(get_lead_times())
                        indices = lead_times - 1
                        return yhat[indices]
        except:
                return [nan for _ in range(len(get_lead_times()))]
            


# In[156]:


# autoregression forecast
from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import array
from numpy import nanmedian
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from warnings import catch_warnings
from warnings import filterwarnings
 
# split the dataset by 'chunkID', return a list of chunks
def to_chunks(values, chunk_ix=0):
        chunks = list()
        # get the unique chunk ids
        chunk_ids = unique(values[:, chunk_ix])
        # group rows by chunk id
        for chunk_id in chunk_ids:
                selection = values[:, chunk_ix] == chunk_id
                chunks.append(values[selection, :])
        return chunks
 
    # return a list of relative forecast lead times
def get_lead_times():
        return [1, 2, 3, 4, 5, 10, 17, 24, 48, 72]
 
    # interpolate series of hours (in place) in 24 hour time
def interpolate_hours(hours):
        # find the first hour
        ix = -1
        for i in range(len(hours)):
                if not isnan(hours[i]):
                        ix = i
                        break
        # fill-forward
        hour = hours[ix]
        for i in range(ix+1, len(hours)):
                # increment hour
                hour += 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
        # fill-backward
        hour = hours[ix]
        for i in range(ix-1, -1, -1):
                # decrement hour
                hour -= 1
                # check for a fill
                if isnan(hours[i]):
                        hours[i] = hour % 24
 
    # return true if the array has any non-nan values
def has_data(data):
        return count_nonzero(isnan(data)) < len(data)
 
    # impute missing data
def impute_missing(rows, hours, series, col_ix):
        # impute missing using the median value for hour in the series
        imputed = list()
        for i in range(len(series)):
                if isnan(series[i]):
                        # get all rows with the same hour
                        matches = rows[rows[:,2]==hours[i]]
                        # fill with median value
                        value = nanmedian(matches[:, col_ix])
                        if isnan(value):
                                value = 0.0
                        imputed.append(value)
                else:
                        imputed.append(series[i])
        return imputed
 
    # layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=5*24):
        # lay out whole series
        data = [nan for _ in range(n_steps)]
        # mark all available data
        for i in range(len(chunk_train)):
                # get position in chunk
                position = int(chunk_train[i, 1] - 1)
                # store data
                data[position] = chunk_train[i, col_ix]
        return data
 
    # fit AR model and generate a forecast
def fit_and_forecast(series):
        # define the model
        model = ARIMA(series, order=(1,0,0))
        # return a nan forecast in case of exception
        try:
                # ignore statsmodels warnings
                with catch_warnings():
                    filterwarnings("ignore")
                        # fit the model
                    model_fit = model.fit()
                        # forecast 72 hours
                    yhat = model_fit.predict(len(series), len(series)+72)
                    # extract lead times
                    lead_times = array(get_lead_times())
                    indices = lead_times - 1
                    return yhat[indices]
        except:
                return [nan for _ in range(len(get_lead_times()))]
 
    # forecast all lead times for one variable
def forecast_variable(hours, chunk_train, chunk_test, lead_times, target_ix):
        # convert target number into column number
        col_ix = 3 + target_ix
        # check for no data
        if not has_data(chunk_train[:, col_ix]):
                forecast = [nan for _ in range(len(lead_times))]
                return forecast
        # get series
        series = variable_to_series(chunk_train, col_ix)
        # impute
        imputed = impute_missing(chunk_train, hours, series, col_ix)
        # fit AR model and forecast
        forecast = fit_and_forecast(imputed)
        return forecast
 
    # forecast for each chunk, returns [chunk][variable][time]
def forecast_chunks(train_chunks, test_input):
        lead_times = get_lead_times()
        predictions = list()
        # enumerate chunks to forecast
        for i in range(len(train_chunks)):
                # prepare sequence of hours for the chunk
                hours = variable_to_series(train_chunks[i], 2)
                # interpolate hours
                interpolate_hours(hours)
                # enumerate targets for chunk
                chunk_predictions = list()
                for j in range(39):
                        yhat = forecast_variable(hours, train_chunks[i], test_input[i], lead_times, j)
                        chunk_predictions.append(yhat)
                chunk_predictions = array(chunk_predictions)
                predictions.append(chunk_predictions)
        return array(predictions)
 
    # convert the test dataset in chunks to [chunk][variable][time] format
def prepare_test_forecasts(test_chunks):
        predictions = list()
        # enumerate chunks to forecast
        for rows in test_chunks:
                # enumerate targets for chunk
                chunk_predictions = list()
                for j in range(3, rows.shape[1]):
                        yhat = rows[:, j]
                        chunk_predictions.append(yhat)
                chunk_predictions = array(chunk_predictions)
                predictions.append(chunk_predictions)
        return array(predictions)
 
    # calculate the error between an actual and predicted value
def calculate_error(actual, predicted):
        # give the full actual value if predicted is nan
        if isnan(predicted):
                return abs(actual)
        # calculate abs difference
        return abs(actual - predicted)
 
    # evaluate a forecast in the format [chunk][variable][time]
def evaluate_forecasts(predictions, testset):
        lead_times = get_lead_times()
        total_mae, times_mae = 0.0, [0.0 for _ in range(len(lead_times))]
        total_c, times_c = 0, [0 for _ in range(len(lead_times))]
        # enumerate test chunks
        for i in range(len(test_chunks)):
                # convert to forecasts
                actual = testset[i]
                predicted = predictions[i]
                # enumerate target variables
                for j in range(predicted.shape[0]):
                    # enumerate lead times
                    for k in range(len(lead_times)):
                            # skip if actual in nan
                            if isnan(actual[j, k]):
                                    continue
                            # calculate error
                            error = calculate_error(actual[j, k], predicted[j, k])
                            # update statistics
                            total_mae += error
                            times_mae[k] += error
                            total_c += 1
                            times_c[k] += 1
        # normalize summed absolute errors
        total_mae /= total_c
        times_mae = [times_mae[i]/times_c[i] for i in range(len(times_mae))]
        return total_mae, times_mae
 
    # summarize scores
def summarize_error(name, total_mae, times_mae):
        # print summary
        lead_times = get_lead_times()
        formatted = ['+%d %.3f' % (lead_times[i], times_mae[i]) for i in range(len(lead_times))]
        s_scores = ', '.join(formatted)
        print('%s: [%.3f MAE] %s' % (name, total_mae, s_scores))
        # plot summary
        pyplot.plot([str(x) for x in lead_times], times_mae, marker='.')
        pyplot.show()
 
    # load dataset
train = loadtxt('/Users/yazmintorresmontana/Documents/Time series/AirQualityPrediction/naive_train.csv', delimiter=',')
test = loadtxt('/Users/yazmintorresmontana/Documents/Time series/AirQualityPrediction/naive_test.csv', delimiter=',')
# group data by chunks
train_chunks = to_chunks(train)
test_chunks = to_chunks(test)
# forecast
test_input = [rows[:, :3] for rows in test_chunks]
forecast = forecast_chunks(train_chunks, test_input)
# evaluate forecast
actual = prepare_test_forecasts(test_chunks)
total_mae, times_mae = evaluate_forecasts(forecast, actual)
# summarize forecast
summarize_error('AR', total_mae, times_mae)


# In[ ]:





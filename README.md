# kaggleweather2016
Autoregressive models for multi-step time series forecasting for a multivariate air pollution time series.
The problem

The Air Quality Prediction dataset describes weather conditions at multiple sites and requires a prediction of air quality measurements over the subsequent three days.

Specifically, weather observations such as temperature, pressure, wind speed, and wind direction are provided hourly for eight days for multiple sites. 


The objective is to predict air quality measurements for the next 3 days at multiple sites. 

The forecast lead times are not contiguous; instead, specific lead times must be forecast over the 72 hour forecast period. They are:


1+1, +2, +3, +4, +5, +10, +17, +24, +48, +72


Further, the dataset is divided into disjoint but contiguous chunks of data, with eight days of data followed by three days that require a forecast.

Not all observations are available at all sites or chunks and not all output variables are available at all sites and chunks. There are large portions of missing data that must be addressed.

The dataset was used as the basis for a short duration machine learning competition (or hackathon) on the Kaggle website in 2012.

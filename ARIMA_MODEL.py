#********Library*************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
inicio=time.time()
# Importing time series specific libraries
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import bartlett
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima.model import ARIMA
import pmdarima
from pmdarima import auto_arima
from statsmodels.tsa.statespace import sarimax
import prophet
from prophet import Prophet
# Miscellaneous libararies
import warnings
warnings.filterwarnings('ignore')
from math import sqrt
from random import random
from Funtion_apply import *
# Libaraies for evaluation of model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error
from statsmodels.tsa.arima.model import ARIMAResults
#***************************Import Data***********************************
df=import_data('C:/Users/milla/Documents/Maestria (Tesis)/Capitulo 2/Modelo Arima de la Demanda/Investigation_Code/international-airline-passengers.csv')
#**************************Verification***********************************
verification(df)
#***************************Processing**********************************
y=processing_data(df)
# print(y)
#************************Plot********************************
graficar(y)
#***********************Histogram****************************
histograma(y)
#**********************Grafica cajas y vigotes***************
boxplot(y)
#**********************descomposicion de la serie de tiempo*******
descomp_time_series(y)
#**********************Grafico de  correlación********************
correlacion(y)
#**********************Statistic analysis*************************
Static_analysis(y)
#*********************Stationary analysis*************************
test_stationarity(y,12)
#************************Series_stationary************************
ts_log=series_stationary_log(y)
#*************************Analisar la estacionalidad del sitema***********
test_stationarity(ts_log,12)
#*************************TRazar la media movil ***************************
mean_mov(12,ts_log)
#**************************Realizar la diferenciacion*********************
ts_log_diff=diff(ts_log)
#*************************Analisar la estacionalidad del sitema***********
test_stationarity(ts_log_diff,12)
#**************************Realizar otra diferenciacion*********************
ts_log_diff2=diff(ts_log_diff)
#*************************Analisar la estacionalidad del sitema***********
test_stationarity(ts_log_diff2,12)
#*************************Grafico de la correlacion **********************
correlacion(ts_log_diff2)
#************************seleccionar las primeras 120 de training********
# Splitting TS data into train and test set for model training and testing
train_ts = ts_log.iloc[0:120,]
test_ts = ts_log.iloc[120: ,]
train_ts
#**************************MODELO ARIMA*********************************
modelo=model_arima(train_ts)
#? esto es lo mismo pero se hace manual
# model = ARIMA(train_ts, order = (0,1,1), seasonal_order=(0,1,1,12))
# results = model.fit()
#********************************Prediccion del modelo********************************
predict(modelo,test_ts,ts_log)

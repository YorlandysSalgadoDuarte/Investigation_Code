#********Library*************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
#*********************************************************************************************************
def import_data(ruta_archivo:str):
    df = pd.read_csv(ruta_archivo, names=['year', 'passengers'], header=0)
    df = df.iloc[:(len(df) - 1),]
    # print(df)
    return df
#*********************************verification*****************************
def verification(df):
    print(df.describe(include = 'all'))
    print(f'Time period start : {df.year.min()}\nTime period end : {df.year.max()}')
    print(df.columns,df.shape)
#**********************************Processing data******************************
def processing_data(df):
    df['year'] = pd.to_datetime(df['year'], format = '%Y-%m')
    y = df.set_index('year')
    y,y.index,y
    y.isnull().sum()
    return y
#**********************************Graficar*************************************
def graficar(y):
    plt.figure(figsize = (15,6))
    plt.plot(y,linewidth = 4)
    plt.xlabel('Month')
    plt.ylabel('No. of air passengers')
    # plt.show()
    #*****************************Histogram************************************
def histograma(y):
    plt.figure(figsize = (15,6))
    sns.histplot(y, kde = True)
    plt.xlabel('No. of passengers')
    plt.ylabel('Count')
    # plt.show()
    #***************************Grafica de cajas y vigotes********************
def boxplot(y):
    plt.figure(figsize = (15,6))
    sns.boxplot(x = y.index.year, y = y.passengers)
    # plt.show()
#**********************descomposicion de la serie de tiempo*******
def descomp_time_series(y):
    from pylab import rcParams
    rcParams['figure.figsize'] = 18,8
    decomposition = sm.tsa.seasonal_decompose(y, model = 'multiplicative')
    plt.figure(figsize = (18,8))
    decomposition.plot()
    # plt.show()
#****************************Grafico de la autocorrelación y la autocorrelación parcial******************
def correlacion(y):
    plt.figure()
    plt.subplot(211)
    plot_acf(y['passengers'], ax=plt.gca(), lags = 30)
    plt.subplot(212)
    plot_pacf(y['passengers'], ax=plt.gca(), lags = 30)
    # plt.show()
    #*******************************Statistic analysis*****************************************************************
def Static_analysis(y):
# Rolling Mean & Rolling Standard Deviation
    rolmean = y.rolling(window = 12).mean() # Calcula con periodo windows la media movil 
    rolstd = y.rolling(window = 12).std() # caclula la desviacino estandar movil segun la  windows 
    plt.figure(figsize = (15,6))
    orig = plt.plot(y, color = 'blue', label ='Original')
    mean  = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std. Dev.')
    plt.legend(loc = 'best')
    # plt.show()
#****************************Stationary analysis*******************************************************
# Creating general function to test stationarity of a time series
def test_stationarity(timeseries,p):
    B = timeseries.iloc[:, 0]
    B = B.reset_index(drop=True)
    # Rolling Mean & Rolling Standard Deviation
    rolmean = timeseries.rolling(window = p).mean()
    rolstd = timeseries.rolling(window = p).std()

    plt.figure(figsize = (15,6))
    orig = plt.plot(timeseries, color = 'blue', label ='Original')
    mean  = plt.plot(rolmean, color = 'red', label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std. Dev.')
    plt.legend(loc = 'best')
    # plt.show()
    # Augmented Dicky-Fuller Test
    print('-------------Results of Dicky Fuller Test -------------')
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(data = dftest[0:4], index = ['Test Statistic : adf', 'p-value : MacKinnon\'s approximate p-value',
                                                     'No. of Lags used', 'No. of observations used'])
    for key,value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    dfoutput['Maximized AIC:'] = dftest[5]
    print(dfoutput)
    if dftest[1]>0.05 :
            print("the null hypothesis is fulfilled for no stationary series ")
    else:
        print("the  hypothesis is fulfilled for  stationary series ")
    #! Agree the Bartlett test
    # TODO se selecciona para la prueba de B todas las filas de la primera columna 
    # Divide the data into three equal parts
    if len(B)//2==0 :
        part_size = len(B) // 2
        seg1 = B[:part_size]
        seg2 = B[2*part_size:]
            # # Apply the Bartlett test to each pair of segments
        resultado_bartlett_seg1_seg2 = bartlett(seg1, seg2)
# # Print the results
        print("Valor p de la prueba de Bartlett para seg1 y seg2:", resultado_bartlett_seg1_seg2.pvalue)
        if resultado_bartlett_seg1_seg2.pvalue < 0.05:
            print('La serie no es estacionaria en VARIANZA')
        else:
            print('La serie es estacionaria en VARIANZA')
    else:
    # Adjust the parts size if needed to make them approximately equal
        part_size = len(B) // 3
        # Divide the series into three parts
        seg1 = B[:part_size]
        seg2 = B[part_size:2*part_size]
        seg3 = B[2*part_size:]
        print(seg3)
        # Apply the Bartlett test to each pair of segments
        resultado_bartlett_seg1_seg2 = bartlett(seg1, seg2)
        resultado_bartlett_seg2_seg3 = bartlett(seg2, seg3)
        resultado_bartlett_seg3_seg1 = bartlett(seg3, seg1)
        # Print the results
        print("Valor p de la prueba de Bartlett para seg1 y seg2:", resultado_bartlett_seg1_seg2.pvalue)
        print("Valor p de la prueba de Bartlett para seg2 y seg3:", resultado_bartlett_seg2_seg3.pvalue)
        print("Valor p de la prueba de Bartlett para seg3 y seg1:", resultado_bartlett_seg3_seg1.pvalue)
        if resultado_bartlett_seg1_seg2.pvalue < 0.05 and resultado_bartlett_seg2_seg3.pvalue < 0.05 and resultado_bartlett_seg3_seg1.pvalue < 0.05:
            print('La serie no es estacionaria en VARIANZA')
        else:
            print('La serie es estacionaria en VARIANZA')
#***********************Making_series_stationary*************************************
def series_stationary_log(y):
    ts_log = np.log(y)
    plt.plot(ts_log)
    return ts_log
#***********************************Trazar la media movil ***********************************
def mean_mov(p,ts_log):
    mov_avg = ts_log.rolling(window =p).mean()
    plt.plot(ts_log, color = 'blue', label ='Log Transformed TS')
    plt.plot(mov_avg, color = 'red', label = 'Moving Avg. (k = 12)')
    plt.legend(loc = 'best')
#**************************Realizar la diferenciacion*******************************
def diff(ts_log) :
    ts_log_diff = ts_log - ts_log.shift(periods = 1) # first differencing
    ts_log_diff
# dropping NA values from differenced TS
    ts_log_diff = ts_log_diff.dropna()
    ts_log_diff
    plt.plot(ts_log_diff)
    return  ts_log_diff
#********************************MODELO ARIMA **********************************************

# Auto arima: seleccion basada en AIC
# ==============================================================================
def model_arima(train_ts):
    modelo = auto_arima(
                y                 = train_ts,
                start_p           = 0,
                start_q           = 0,
                max_p             = 3,
                max_q             = 3,
                seasonal          = True,
                test              = 'adf',
                m                 = 12, # periodicidad de la estacionalidad
                d                 = None, # El algoritmo determina 'd'
                D                 = None, # El algoritmo determina 'D'
                trace             = True,
                error_action      = 'ignore',
                suppress_warnings = True,
                stepwise          = True,
                random_state      =4,
                n_fits            =15000
    )
    print(modelo.summary())
    return modelo
#********************************Prediccion del modelo********************************
def predict(modelo,test_ts,ts_log):
# Obtén las predicciones para el conjunto de prueba
    predictions = modelo.predict(n_periods=120)
    predictions_test_=modelo.predict(len(test_ts))
    # Crea un DataFrame con las fechas y las predicciones
    predictions_df = pd.DataFrame(predictions, index=test_ts.index, columns=['Predictions'])
    # # Imprime el DataFrame con las predicciones
    print(predictions_df)
    predictions_df.describe()
    plt.figure(figsize = (15,6))
    plt.plot(ts_log, color = 'green', label = 'Log Transformed Original data')
    plt.plot(predictions_df, color = 'blue', label = 'Predicted values for train dataset')
    plt.plot(predictions, color = 'orange', label = 'Predicted values for test dataset')
    plt.xlabel('Month')
    plt.ylabel('Log of no. of passengers')
    plt.title('ARIMA(3,2,2)(0,1,0)[12] qualitative performance')
    plt.legend(loc = 'best')
    # plt.show()
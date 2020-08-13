import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

def arima():
    # change later
    bitcoin_data = pd.read_csv('C:\\Users\\alexg\\Desktop\\TCF\\Data\\bitcoin_data.csv')
    bitcoin_data = bitcoin_data.drop(labels=['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1)
    history = bitcoin_data.to_numpy()
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
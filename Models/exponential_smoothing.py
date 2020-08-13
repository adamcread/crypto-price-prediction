import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def exponential_smoothing():
    bitcoin_data = pd.read_csv('C:\\Users\\alexg\\Desktop\\TCF\\Data\\bitcoin_data.csv')
    bitcoin_data = bitcoin_data['Close']
    fit4 = Holt(bitcoin_data, damped=True, exponential=True).fit()
    fcast4 = fit4.predict(start=len(bitcoin_data), end=len(bitcoin_data)).rename()
    # type of variable prediction is numpy.float64
    prediction = fcast4[len(bitcoin_data)]

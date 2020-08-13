import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

bitcoin_data = pd.read_csv('Data/bitcoin_data.csv')
bitcoin_data = bitcoin_data['Close']

print(bitcoin_data)
train_data = bitcoin_data[:2500]
test_data = bitcoin_data[2500:]

print(test_data)


fit1 = SimpleExpSmoothing(train_data).fit()
fcast1 = fit1.predict(start=2490, end=2500).rename()
print(fcast1)

fit2 = Holt(train_data, damped=True).fit()
fcast2 = fit2.predict(start=2490, end=2500).rename()
fit3 = Holt(train_data, exponential=True).fit()
fcast3 = fit3.predict(start=2490, end=2500).rename()
fit4 = Holt(train_data, damped=True, exponential=True).fit()
fcast4 = fit4.predict(start=2490, end=2500).rename()


train_data[2490:].plot()
test_data.head(n=1).plot(marker='o', label="test value", color="purple", legend=True)
fcast1.plot(legend=True, marker='x', label='alpha={}'.format(fit1.model.params['smoothing_level']))
fcast2.plot(legend=True, marker='x', label="Additive damped trend")
fcast3.plot(legend=True, marker='x', label="Additive exponential trend")
fcast4.plot(legend=True, marker='x', label="Additive exponential and damped trend")
# fit1.fittedvalues.plot(ax=ax, color='blue')

plt.show()

# data = [446.6565,  454.4733,  455.663 ,  423.6322,  456.2713,  440.5881, 425.3325,  485.1494,  506.0482,  526.792 ,  514.2689,  494.211 ]
# index= pd.date_range(start='1996', end='2008', freq='A')
# oildata = pd.Series(data, index)
# print(oildata)

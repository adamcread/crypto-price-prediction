import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

bitcoin_data = pd.read_csv('Data/bitcoin_data.csv')
bitcoin_data = bitcoin_data.drop(labels=['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap'], axis=1)

train_data = bitcoin_data[:2500]
test_data = bitcoin_data[2500:].reset_index(drop=True)
history = train_data.to_numpy()
print(test_data)
predictions = list()

for d in test_data.to_numpy():
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    # print(model_fit.summary())
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = d[0]
    # print(obs)
    history = np.append(history, [obs])
    print('predicted=%f, expected=%f' % (yhat, float(obs)))


plt.plot(test_data)
plt.plot(predictions, color='red')
plt.show()
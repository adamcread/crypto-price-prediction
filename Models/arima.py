import pandas as pd
from datetime import date
from statsmodels.tsa.arima_model import ARIMA
from pypfopt.discrete_allocation import DiscreteAllocation

# get predictions from ARIMA model
def model_prediction(data): 
    # train model and fit to data
    model = ARIMA(data, order=(5,1,0))
    model_fit = model.fit(disp=0)

    # make prediction about next day closing price
    output = model_fit.forecast()
    predicted_value = output[0]

    return predicted_value

# boilerplate trading bot plug in model predictions
def trading_bot(event=None, context=None):
    # get historical crypto data from gbq (sort by date to ensure data is in correct order)
    historical_data_query = """
        SELECT *
        FROM `crypto-prediction-286314.crypto_predictor.historical_data`
    """
    historical_data = pd.read_gbq(historical_data_query).sort_values(by='date').reset_index(drop=True)

    # initialise variables to store expected returns and predicted values
    expected_returns = pd.Series(dtype='float64')
    predicted = pd.DataFrame()

    # list currencies to trade with
    currencies = ["bitcoin", "ethereum", "xrp", "litecoin", "dash"]
    for currency in currencies:
        # train model on historical data
        crypto_data = historical_data['{}Close'.format(currency)].to_numpy()

        predicted_value = model_prediction(crypto_data)
        closing_value = crypto_data[-1]
        
        # store predicted value and expected return (predicted percentage increase)
        predicted['{}Predicted'.format(currency)] = [predicted_value]
        expected_returns['{}Close'.format(currency)] = (predicted_value-closing_value) / closing_value

    # get portfolio data from gbq (sort by date to ensure data is in correct order)
    pf_query = """
            SELECT *
            FROM `crypto-prediction-286314.crypto_predictor.arima_portfolio`
    """
    pf_data = pd.read_gbq(pf_query).sort_values(by='date').reset_index(drop=True)

    # obtain most recent closing prices and portfolio
    latest_prices = historical_data.drop('date', axis=1).iloc[-1]
    pf_latest = pf_data.iloc[-1]

    # calculate updated portfolio value
    pf_value = sum([pf_latest[c+'Bought']*latest_prices[c+'Close'] for c in currencies]) + float(pf_latest['pfUnallocated'])

    # calculate weights as a ratio of positive the expected return of the stock is predicted to be
    # if no positive returns all weights will be set to 0 and a very small amount of the portfolio will be allocated
    positive_returns = sum([er for er in expected_returns if er > 0])
    weights = [max(0, x/positive_returns) for x in expected_returns] if positive_returns else [0]*len(expected_returns)
    cleaned_weights = {expected_returns.index[i]: float(weights[i]) for i in range(len(expected_returns))}

    # use weightings to calculate how many coins to purchase
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=pf_value)
    allocation, unallocated = da.lp_portfolio()

    # for each coin store tomorrows's predicted, today's close, coins bought
    # also store current portfolio value + unallocated money (to help calculate updated portfolio value)
    pf_updated = {'date': date.today().strftime("%Y-%m-%d"), 'pfValue': pf_value, 'pfUnallocated': unallocated}
    for currency in currencies:
        pf_updated['{}Close'.format(currency)] = float(latest_prices['{}Close'.format(currency)])
        pf_updated['{}Predicted'.format(currency)] = float(predicted['{}Predicted'.format(currency)])
        pf_updated['{}Bought'.format(currency)] = float(allocation.get('{}Close'.format(currency), 0))

    # put new portfolio into dataframe and insert today's date
    pf_updated = pd.DataFrame(pf_updated, index=[0])

    # add new portfolio to existing portfolio data and push to gbq
    pf_data = pd.concat([pf_data, pf_updated], ignore_index=True)

    pf_data.to_gbq(destination_table='crypto_predictor.arima_portfolio',
                project_id="crypto-prediction-286314",
                if_exists='replace')

trading_bot()
import pandas as pd
from statsmodels.tsa.api import Holt
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from datetime import date

def exp_smoothing(event=None, context=None):
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
        model_fit = Holt(crypto_data, damped=True, exponential=True).fit()

        # make prediction about next day closing price
        predicted_value = float(model_fit.predict())
        closing_value = crypto_data[-1]

        # store predicted value and expected return (predicted percentage increase)
        predicted['{}Predicted'.format(currency)] = [predicted_value]
        expected_returns['{}Close'.format(currency)] = (predicted_value-closing_value) / closing_value

    # get portfolio data from gbq (sort by date to ensure data is in correct order)
    pf_query = """
            SELECT *
            FROM `crypto-prediction-286314.crypto_predictor.expSmoothing_portfolio`
    """
    pf_data = pd.read_gbq(pf_query).sort_values(by='date').reset_index(drop=True)

    # obtain most recent closing prices and portfolio
    latest_prices = historical_data.drop('date', axis=1).iloc[-1]
    pf_latest = pf_data.iloc[-1]

    # calculate updated portfolio value
    pf_value = sum([pf_latest[c+'Bought']*latest_prices[c+'Close'] for c in currencies]) + float(pf_latest['pfUnallocated'])

    # use efficient frontier to obtain weightings of coins to purchase
    # drop date as it does not need to be included in covariance matrix
    S = risk_models.risk_matrix(historical_data.drop('date', axis=1))
    ef = EfficientFrontier(expected_returns, S, gamma=5)
    ef.min_volatility()
    cleaned_weights = ef.clean_weights()

    # use weightings to calculate how many coins to purchase
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=pf_value)
    allocation, unallocated = da.lp_portfolio()

    # for each coin store tomorrows's predicted, today's close, coins bought
    # also store current portfolio value + unallocated money (to help calculate updated portfolio value)
    pf_updated = {}
    for currency in currencies:
        pf_updated['{}Close'.format(currency)] = float(latest_prices['{}Close'.format(currency)])
        pf_updated['{}Predicted'.format(currency)] = float(predicted['{}Predicted'.format(currency)])
        pf_updated['{}Bought'.format(currency)] = float(allocation.get('{}Close'.format(currency), 0))
    pf_updated['pfValue'] = pf_value
    pf_updated['pfUnallocated'] = unallocated

    # put new portfolio into dataframe and insert today's date
    pf_updated = pd.DataFrame(pf_updated, index=[0])
    pf_updated.insert(0, 'date', date.today().strftime("%Y-%m-%d"))

    # add new portfolio to existing portfolio data and push to gbq
    pf_data = pd.concat([pf_data, pf_updated], ignore_index=True)

    pf_data.to_gbq(destination_table='crypto_predictor.expSmoothing_portfolio',
                project_id="crypto-prediction-286314",
                if_exists='replace')

exp_smoothing()
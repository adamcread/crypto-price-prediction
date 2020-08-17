from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import date

def get_historical_data(event=None, context=None):
    # list currencies to trade with
    currencies = ["bitcoin", "ethereum", "xrp", "litecoin", "dash"]

    close_data = pd.DataFrame()
    for currency in currencies:
        # get date and make url to scrape from
        end_date = date.today().strftime("%Y%m%d")
        url = "https://coinmarketcap.com/currencies/{}/historical-data/?start=20150807&end={}".format(currency, end_date)

        # scrape html from url
        content = requests.get(url).content
        soup = BeautifulSoup(content, 'html.parser')

        # select the bitcoin historical table and extract data
        btc_table = soup.find_all('table')[2]
        data = [[td.text.strip() for td in tr.find_all('td')] for tr in btc_table.find('tbody').find_all('tr')]

        # move data into df
        currency_data = pd.DataFrame(data)
        currency_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap']

        # convert numerical columns into numerical data
        # commas removed as if not data > 999 will be stored as string
        close_data['{}Close'.format(currency)] = pd.to_numeric(currency_data['Close'].str.replace(",", ""))
    
    # add date so when read we can sort and ensure data is in correct order
    close_data['date'] = pd.to_datetime(currency_data['Date'].str.replace(",", ""))

    # upload df to gbq
    close_data.to_gbq(destination_table='crypto_predictor.historical_data',
            project_id="crypto-prediction-286314",
            if_exists='replace')

get_historical_data()
from bs4 import BeautifulSoup
from google.cloud import bigquery
import pandas_gbq
import requests
import pandas as pd
from datetime import date


def get_bitcoin_data(event=None, context=None):
    # get date and make url to scrape from
    end_date = date.today().strftime("%Y%m%d")
    url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130429&end={}".format(end_date)

    # scrape html from url
    content = requests.get(url).content
    soup = BeautifulSoup(content, 'html.parser')

    # select the bitcoin historical table and extract data
    btc_table = soup.find_all('table')[2]
    data = [[td.text.strip() for td in tr.find_all('td')] for tr in btc_table.find('tbody').find_all('tr')]

    # move data into df
    historical_data = pd.DataFrame(data)
    historical_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap']

    # convert numerical columns into numerical data
    # commas removed as if not data > 999 will be stored as string
    for col in historical_data.columns[1:]:
        historical_data[col] = pd.to_numeric(historical_data[col].str.replace(",", ""))

    # upload df to gbq
    historical_data.to_gbq(destination_table='crypto_predictor.btc', 
            project_id="crypto-prediction-286314",
            if_exists='replace')
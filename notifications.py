import requests
import pandas as pd

def notification(event=None, content=None):
    models = ['arima', 'expSmoothing', 'tbats_model']
    for model in models:
        pf_query = """
                    SELECT *
                    FROM `crypto-prediction-286314.crypto_predictor.{}_portfolio`
            """.format(model)
        pf_data = pd.read_gbq(pf_query).sort_values(by='date').reset_index(drop=True).iloc[-1]

        string_pf = "MODEL NAME: {}\n".format(model) + pf_data.to_string()

        requests.post(
            url="https://slack.com/api/chat.postMessage",
            data= {
                "token": "xoxp-1309063605814-1339633930336-1316278098259-3be9d3bfa3fa2ecd678cc6924a51b46c",
                "channel": "#{}".format(model.lower()),
                "text": string_pf
            }
        )

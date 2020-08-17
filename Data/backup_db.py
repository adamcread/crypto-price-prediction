import pandas as pd

def backup_database(event=None, context=None):
    # models currently running needing to be backed up
    models = ['arima', 'expSmoothing']

    for model in models:
        # get live portfolio
        model_query = """
            SELECT *
            FROM `crypto-prediction-286314.crypto_predictor.{}_portfolio`
        """.format(model)
        model_data = pd.read_gbq(model_query).sort_values(by='date').reset_index(drop=True)

        # get backup portfolio
        backup_query = """
            SELECT *
            FROM `crypto-prediction-286314.backup.{}_portfolio`
        """.format(model)
        backup_data = pd.read_gbq(backup_query).sort_values(by='date').reset_index(drop=True)

        # add new portfolio to existing portfolio data and push to gbq
        pf_data = backup_data.append(model_data.iloc[-1], ignore_index=True)
        pf_data.to_gbq(destination_table='backup.{}_portfolio'.format(model),
                project_id="crypto-prediction-286314",
                if_exists='replace')

backup_database()
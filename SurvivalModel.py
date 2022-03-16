import streamlit as st
import pandas as pd
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
from sklearn.metrics import classification_report

train_data = pd.read_csv('veteran.csv')


def eval(actual, all_predictions):
    cm = classification_report(actual, all_predictions, output_dict=True)
    return pd.DataFrame(cm).transpose()


# get the model that uses the message data
def get_model(algorithm):

    if algorithm == 'Cox Proportional-Hazards Model':
        spam_detect_model = CoxPHFitter()
    else:
        spam_detect_model = WeibullAFTFitter()

    spam_detect_model.fit(train_data, duration_col='time', event_col='status')

    return spam_detect_model


# predict if a message is ham or spam
def predict_suvival(df, algorithm_selected):

    model = get_model(algorithm)

    prediction = model.predict(df)

    return prediction

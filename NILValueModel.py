import streamlit as st
import pandas as pd
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

train_data = pd.read_csv('veteran.csv')


def fill_celltype_matrix(df, cell_selected):
    if cell_selected == 'Large':
        df.at[0, 'celltype_large'] = 1
        df.at[0, 'celltype_smallcell'] = 0
        df.at[0, 'celltype_squamous'] = 0
    elif cell_selected == 'Smallcell':
        df.at[0, 'celltype_large'] = 0
        df.at[0, 'celltype_smallcell'] = 1
        df.at[0, 'celltype_squamous'] = 0
    elif cell_selected == 'Squamous':
        df.at[0, 'celltype_large'] = 0
        df.at[0, 'celltype_smallcell'] = 0
        df.at[0, 'celltype_squamous'] = 1
    else:
        df.at[0, 'celltype_large'] = 0
        df.at[0, 'celltype_smallcell'] = 0
        df.at[0, 'celltype_squamous'] = 0
    return df


# transform the data to all numeric types
def transform_train_data(df):
    df_cat = pd.get_dummies(df[['celltype']], drop_first=True)
    df_transformed = df.join(df_cat)
    df_transformed = df_transformed.drop(columns=['celltype', 'Unnamed: 0'])
    return df_transformed


def get_model(algorithm):
    if algorithm == 'Cox Proportional-Hazards Model':
        spam_detect_model = CoxPHFitter()
    else:
        spam_detect_model = WeibullAFTFitter()

    train_df = transform_train_data(train_data)
    spam_detect_model.fit(train_df, duration_col='time', event_col='status')

    return spam_detect_model


# map the treatment type to 0,1, map prior therapy to 0,1
def clean_trt_prior(df):
    df.trt = df.trt.map({'Standard' : 0, 'Test' : 1})

    df.prior = df.prior.map({'No': 0, 'Yes': 1})
    return df


def predict_suvival(df, cell_selected, algorithm):
    # filling in the cell type matrix given selection
    df = fill_celltype_matrix(df, cell_selected)

    df = clean_trt_prior(df)

    model = get_model(algorithm)

    prediction = model.predict_expectation(df)

    return prediction

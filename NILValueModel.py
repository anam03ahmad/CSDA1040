import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from SchoolHometownGroups import get_hometown_count_group, get_hometown_max_group, get_school_max_group, get_school_count_group

train_df = pd.read_csv('nil_value_data.csv')
norm_df = pd.read_csv('nil_value_not_normalized.csv')

train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='Position')))]
train_df = train_df[train_df.columns.drop(list(train_df.filter(regex='Sport')))]


def normalize(df):

    to_norm_df = df.append(norm_df, ignore_index=True)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(to_norm_df)
    to_norm_df.loc[:, :] = scaled_values

    return to_norm_df.head(1)


def school_hometown_mapping(df, school_selected, hometown_selected):

    df['School Max Group'] = get_school_max_group(school_selected)
    df['School Count Group'] = get_school_count_group(school_selected)

    df['Hometown Max Group'] = get_hometown_max_group(hometown_selected)
    df['Hometown Count Group'] = get_hometown_count_group(hometown_selected)

    df = df.drop(['School', 'Hometown'], axis=1)
    return df


def get_model(algorithm):
    if algorithm == 'Random Forest Model':
        nil_prediction_model = RandomForestRegressor()
    else:
        nil_prediction_model = XGBRegressor()

    cols = train_df.columns.tolist()
    cols.remove('NIL Value')
    x_cols = cols

    nil_prediction_model.fit(train_df[x_cols], train_df['NIL Value'].ravel())

    return nil_prediction_model


def predict_nil(df, hometown_selected, school_selected, algorithm):
    # filling in the cell type matrix given selection

    print('original', df)

    df = school_hometown_mapping(df, school_selected, hometown_selected)

    print('after mapping', df)

    df = normalize(df)

    print('after normalizing', df)

    model = get_model(algorithm)

    prediction = model.predict(df)

    return prediction

import pandas as pd

hometown_df = pd.read_csv('hometown_groups.csv')
school_df = pd.read_csv('school_groups.csv')


def get_hometown_count_group(hometown):
    return hometown_df[hometown_df['Hometown'] == hometown]['Hometown Count Group']


def get_hometown_max_group(hometown):
    return hometown_df[hometown_df['Hometown'] == hometown]['Hometown Max Group']


def get_hometowns():
    return hometown_df.Hometown.tolist()


def get_school_count_group(school):
    return school_df[school_df['School'] == school]['School Count Group']


def get_school_max_group(school):
    return school_df[school_df['School'] == school]['School Max Group']


def get_schools():
    return school_df.School.tolist()


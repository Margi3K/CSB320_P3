def clean_data(df):
    return df.fillna(df.median(inplace=True))
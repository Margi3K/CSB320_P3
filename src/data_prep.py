import pandas as pd
from sklearn.preprocessing import StandardScaler

# Remove outliers using the IQR method
def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier columns
    drop_cols = ["Sl_No", "Customer Key"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

# Standardize features
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)

# The full preparation pipeline
def prepare_credit_card_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = remove_outliers_iqr(df)
    df = scale_features(df)
    return df

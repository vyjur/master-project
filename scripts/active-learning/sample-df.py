import os
import pandas as pd

BATCH = 1
PERCENTAGE = 7
PREVIOUS = 2.6

def sample_dataframe(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """
    Returns the first rows of the DataFrame after sorting by 'prob' in ascending order,
    containing the specified percentage of rows.
    
    :param df: Input Pandas DataFrame
    :param percentage: Percentage of rows to select (between 0 and 100)
    :return: Sampled DataFrame
    """
    if 'prob' not in df.columns:
        raise ValueError("DataFrame must contain a 'prob' column for sorting.")
    
    if not (0 < percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    df_sorted = df.sort_values(by='prob', ascending=True)
    print(df_sorted)
    num_rows = int(len(df) * (percentage / 100))
    return df_sorted.head(num_rows)

df = pd.read_csv(f'./data/helsearkiv/batch/dtr/{BATCH}.csv')
print(len(df))
df = df[(df["MedicalEntity"].notna()) & (df["MedicalEntity"] != "O")]
print(df)

final_df = sample_dataframe(df, PERCENTAGE-PREVIOUS)

final_df.to_csv(f'./data/helsearkiv/batch/dtr/{BATCH}-final.csv')

print(len(df), len(final_df), (len(final_df))/len(df)+PREVIOUS/100)
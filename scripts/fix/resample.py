import pandas as pd

file = './data/helsearkiv/batch/dtr/1-final.csv'
df = pd.read_csv(file)
df = df.sort_values(by="prob", ascending=True)  # Sort in ascending order
df = df.iloc[0:4000]
df.to_csv(file)
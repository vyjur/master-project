import pandas as pd

file = './data/helsearkiv/batch/dtr/2.csv'
df = pd.read_csv(file)
df = df.sort_values(by="prob", ascending=True)  # Sort in ascending order
df = df.iloc[0:4000]
save_file = './data/helsearkiv/batch/dtr/2-final.csv'
df.to_csv(save_file)
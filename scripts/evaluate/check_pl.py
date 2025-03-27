import pandas as pd
file = "data/helsearkiv/evaluate/tem1-mer/data/helsearkiv/journal/d1d07d02-8240-4a46-a007-85a69a06def4_0F978AAC-681F-4A61-9F82-0D00316D8FDC/output.csv"
df = pd.read_csv(file)
df['context'] = ''
df.to_csv(file)

import os
import numpy as np
import pandas as pd

folder_path = "./data/helsearkiv/annotated/entity-without-timex/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

for file in entity_files:
    df = pd.read_csv(file, delimiter=',')
    df['TIMEX'] = np.nan
    df.to_csv(file)

            
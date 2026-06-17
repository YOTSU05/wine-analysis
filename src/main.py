import numpy as np
import pandas as pd
import matplotlib

print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)

import pandas as pd

df= pd.read_csv('date/wine.csv')

print(df.info())
print(df.describe())
print(df.head())

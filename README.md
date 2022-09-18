# csproject6037
import pandas as pd
import numpy as np
fn = pd.read_csv('newdf.csv')
print(fn)
above_35= fn[fn['Age'] > 35]
print(above_35.head())
print(fn['Age'] > 35)

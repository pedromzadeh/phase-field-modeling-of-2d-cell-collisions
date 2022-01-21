import numpy as np
import json
import sys
import pandas as pd
import matplotlib.pyplot as plt

path = 'Simulation Data/{}/tabular_stats.json'.format(sys.argv[1])
with open(path,'r') as jfile:
   data = json.load(jfile)
expanded_table1 = pd.DataFrame(data['Pwin_averaged_table'],columns=['ps', 'st', 'subA', 'dv', 'dca', 'Pwin'])

path = 'Simulation Data/{}/tabular_stats.json'.format(sys.argv[2])
with open(path,'r') as jfile:
   data = json.load(jfile)
expanded_table2 = pd.DataFrame(data['Pwin_averaged_table'],columns=['ps', 'st', 'subA', 'dv', 'dca', 'Pwin'])

dfs = [expanded_table1,expanded_table2]
df = pd.concat(dfs)
print(df)

q = (df + df.shift(-462)) /2
q = q.dropna()
print(q)

X = q['dca']
Y = q['Pwin']

X_half = expanded_table1['dca']
Y_half = expanded_table1['Pwin']

plt.scatter(X,Y)
plt.scatter(X_half,Y_half)
plt.show()
import pandas as pd
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt 
full_data=pd.read_csv("processed_data.csv")
print(full_data.shape)
features,target = full_data.drop(columns='Цена'),full_data['Цена']
corr=features.corrwith(target)
print(corr[corr>0.5].sort_values())
corr.abs().plot(kind='hist')
plt.show()
corr[corr.abs()>0.05].sort_values().plot(kind='barh', figsize=(10, 8))
plt.title("Correlation with target")
plt.show()
na_columns = full_data.columns[full_data.isna().mean()>0]
print(na_columns)
sns.boxplot ( data=full_data.loc [ :, : ], orient='h' )
plt.show ()
missing_values_count = full_data.isnull ().sum ()
print ( missing_values_count )
for column in na_columns:
    full_data[column + "__is_na"] = 0
    full_data.loc[full_data[column].isna(), column + "__is_na"] = 1
print(full_data.shape)
full_data.to_csv('ready_data.csv', index=False)
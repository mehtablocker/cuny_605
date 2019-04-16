import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
from statsmodels.formula.api import ols

df = pd.read_csv('https://raw.githubusercontent.com/mehtablocker/cuny_605/master/data_files/who.csv', encoding = "ISO-8859-1")
df.head()
df.plot.scatter('TotExp', 'LifeExp')

lm_fit = ols('LifeExp ~ TotExp', data = df).fit()
lm_fit.summary()

df = df.assign(TotExp_trans = df.TotExp**0.06, LifeExp_trans = df.LifeExp**4.6)
df.plot.scatter('TotExp_trans', 'LifeExp_trans')

lm_fit = ols('LifeExp_trans ~ TotExp_trans', data = df).fit()
lm_fit.summary()
new_df = pd.DataFrame(np.array([1.5, 2.5]), columns = ['TotExp_trans'])
y_bars = lm_fit.predict(new_df)**(1/4.6)
print(y_bars)

lm_fit = ols('LifeExp ~ PropMD + TotExp + PropMD*TotExp', data = df).fit()
lm_fit.summary()
new_df = pd.DataFrame({'PropMD':[0.03], 'TotExp':[14]})
y_bars = lm_fit.predict(new_df)
print(y_bars)

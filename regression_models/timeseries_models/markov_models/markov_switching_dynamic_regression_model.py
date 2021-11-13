import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

#Load the PCE and UMCSENT datasets
df = pd.read_csv(filepath_or_buffer='UMCSENT_PCE.csv', header=0, index_col=0,
                 infer_datetime_format=True, parse_dates=['DATE'])
#Set the index frequency to 'Month-Start'
df = df.asfreq('MS')

#Plot both time series
fig = plt.figure()
fig.suptitle('% Chg in Personal Consumption Expenditure')
df['PCE_CHG'].plot()
plt.show()
fig = plt.figure()
fig.suptitle('% Chg in U. Michigan Consumer Sentiment Index')
df['UMCSENT_CHG'].plot()
plt.show()

#build and train the MSDR model
msdr_model = sm.tsa.MarkovRegression(endog=df['PCE_CHG'], k_regimes=2,
    trend='c', exog=df['UMCSENT_CHG'], switching_variance=True)
msdr_model_results = msdr_model.fit(iter=1000)

#print model training summary
print(msdr_model_results.summary())

df_r = pd.read_csv('JHDUSRGDPBR.csv', header=0, index_col=0, 
                   infer_datetime_format=True, parse_dates=['DATE'])

fig, axes = plt.subplots(3)
ax = axes[0]
ax.plot(df.index, df['PCE_CHG'])
ax.set(title="% Chg in Personal Consumption Expenditure")

ax = axes[1]
ax.plot(df.index, msdr_model_results.smoothed_marginal_probabilities[0])
ax.set(title="Smoothed probability of regime 0")

ax = axes[2]
ax.plot(df.index, msdr_model_results.smoothed_marginal_probabilities[1])
ax.plot(df_r.index, df_r['JHDUSRGDPBR'])
ax.set(title="Smoothed probability of regime 1 super-imposed on GDP based "
             "recession indicator (Orange)")

plt.show()

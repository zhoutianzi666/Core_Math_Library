import numpy as np
import matplotlib.pyplot as plt

import tushare as ts

df = ts.get_today_ticks('601022')
print(df[100:])

n = len(df)

sell = 0
buy = 0
result_sell = np.zeros(n)
result_buy = np.zeros(n)

for i in range(n):
    if(df['type'][i]=="卖出"):
        sell = sell + df['vol'][i]
    if(df['type'][i]=="买入"):
        buy = buy + df['vol'][i]
    result_sell[i] = sell
    result_buy[i] = buy
plt.plot(result_sell, result_buy) 
plt.plot(result_sell, result_sell)
plt.savefig('1.png')


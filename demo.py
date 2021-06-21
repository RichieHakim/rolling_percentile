# from rolling_percentile.rolling_percentile import rolling_percentile, initialize_rp
from .rolling_percentile import rolling_percentile, initialize_rp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

x_in = np.random.rand(1000,10000)

window_length = 1001 # must be odd integer
percentile = 10 # 0-100

initialize_rp() # not necessary, just used to compile all the jit functions and store in cache
tic = time()
x_out_rp = rolling_percentile(x_in, win_len=window_length, ptile=percentile)
print(f'rolling_percentile computation time: {time() - tic} seconds')

x_in_df = pd.DataFrame(x_in)
tic = time()
x_out_pd = x_in_df.rolling(window=window_length, center=True, axis=1).quantile(quantile=(percentile/100))
print(f'pandas computation time: {time() - tic} seconds')
x_out_pd = np.array(x_out_pd)

win_half_len = window_length//2
outputs_equivalent_check = np.allclose(x_out_rp[:,win_half_len:-(win_half_len+1)], x_out_pd[:,win_half_len:-(win_half_len+1)])
print(f'Outputs from rolling_percentile and pandas are exactly the same: {outputs_equivalent_check}')

plt.figure()
plt.plot(x_out_rp[0])
plt.plot(x_out_pd[0])
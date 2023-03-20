import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from head_shoulders import find_hs_patterns, get_pattern_return
    

data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

data = np.log(data)
dat_slice = data['close'].to_numpy()


orders = list(range(1, 49))
ihs_count = []
ihs_early_count = []
hs_count = []
hs_early_count = []

ihs_wr = []
ihs_early_wr = []
hs_wr = []
hs_early_wr = []

ihs_wr_stop = []
ihs_early_wr_stop = []
hs_wr_stop = []
hs_early_wr_stop = []

ihs_avg = []
ihs_early_avg = []
hs_avg = []
hs_early_avg = []

ihs_avg_stop = []
ihs_early_avg_stop = []
hs_avg_stop = []
hs_early_avg_stop = []


ihs_total_ret = []
ihs_early_total_ret = []
hs_total_ret = []
hs_early_total_ret = []

ihs_total_ret_stop = []
ihs_early_total_ret_stop = []
hs_total_ret_stop = []
hs_early_total_ret_stop = []


for order in orders:
    hs_patterns, ihs_patterns = find_hs_patterns(dat_slice, order, False)
    hs_patterns_early, ihs_patterns_early = find_hs_patterns(dat_slice, order, True)
    
    hs_df = pd.DataFrame()
    ihs_df = pd.DataFrame()
    hs_early_df = pd.DataFrame()
    ihs_early_df = pd.DataFrame()

    # Load pattern attributes into dataframe
    for i, hs in enumerate(hs_patterns):
        hs_df.loc[i, 'head_width'] = hs.head_width
        hs_df.loc[i, 'head_height'] = hs.head_height
        hs_df.loc[i, 'r2'] = hs.pattern_r2
        hs_df.loc[i, 'neck_slope'] = hs.neck_slope
        
        hp = int(hs.head_width)
        if hs.break_i + hp >= len(data):
            hs_df.loc[i, 'hold_return'] = np.nan
        else:
            ret = -1 * (dat_slice[hs.break_i + hp] - dat_slice[hs.break_i])
            hs_df.loc[i, 'hold_return'] = ret 
        
        hs_df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, hs) 
    
    for i, hs in enumerate(ihs_patterns):
        ihs_df.loc[i, 'head_width'] = hs.head_width
        ihs_df.loc[i, 'head_height'] = hs.head_height
        ihs_df.loc[i, 'r2'] = hs.pattern_r2
        ihs_df.loc[i, 'neck_slope'] = hs.neck_slope
        
        hp = int(hs.head_width)
        if hs.break_i + hp >= len(data):
            ihs_df.loc[i, 'hold_return'] = np.nan
        else:
            ret = dat_slice[hs.break_i + hp] - dat_slice[hs.break_i]
            ihs_df.loc[i, 'hold_return'] = ret 
        
        ihs_df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, hs) 
    
    for i, hs_early in enumerate(hs_patterns_early):
        hs_early_df.loc[i, 'head_width'] = hs_early.head_width
        hs_early_df.loc[i, 'head_height'] = hs_early.head_height
        hs_early_df.loc[i, 'r2'] = hs_early.pattern_r2
        hs_early_df.loc[i, 'neck_slope'] = hs_early.neck_slope
        
        hp = int(hs_early.head_width)
        if hs_early.break_i + hp >= len(data):
            hs_early_df.loc[i, 'hold_return'] = np.nan
        else:
            ret = -1 * (dat_slice[hs_early.break_i + hp] - dat_slice[hs_early.break_i])
            hs_early_df.loc[i, 'hold_return'] = ret 
        
        hs_early_df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, hs_early) 
    
    # Load pattern attributes into dataframe
    for i, hs_early in enumerate(ihs_patterns_early):
        ihs_early_df.loc[i, 'head_width'] = hs_early.head_width
        ihs_early_df.loc[i, 'head_height'] = hs_early.head_height
        ihs_early_df.loc[i, 'r2'] = hs_early.pattern_r2
        ihs_early_df.loc[i, 'neck_slope'] = hs_early.neck_slope
        
        hp = int(hs_early.head_width)
        if hs_early.break_i + hp >= len(data):
            ihs_early_df.loc[i, 'hold_return'] = np.nan
        else:
            ret = dat_slice[hs_early.break_i + hp] - dat_slice[hs_early.break_i]
            ihs_early_df.loc[i, 'hold_return'] = ret 
        
        ihs_early_df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, hs_early) 
    
    if len(ihs_df) > 0:
        ihs_count.append(len(ihs_df))
        ihs_avg.append(ihs_df['hold_return'].mean())
        ihs_wr.append(len(ihs_df[ihs_df['hold_return'] > 0]) / len(ihs_df))
        ihs_total_ret.append(ihs_df['hold_return'].sum())
        
        ihs_avg_stop.append(ihs_df['stop_return'].mean())
        ihs_wr_stop.append(len(ihs_df[ihs_df['stop_return'] > 0]) / len(ihs_df))
        ihs_total_ret_stop.append(ihs_df['stop_return'].sum())
    else:
        ihs_count.append(0)
        ihs_avg.append(np.nan)
        ihs_wr.append(np.nan)
        ihs_total_ret.append(0)
        
        ihs_avg_stop.append(np.nan)
        ihs_wr_stop.append(np.nan)
        ihs_total_ret_stop.append(0)
    
    if len(hs_df) > 0:
        hs_count.append(len(hs_df))
        hs_avg.append(hs_df['hold_return'].mean())
        hs_wr.append(len(hs_df[hs_df['hold_return'] > 0]) / len(hs_df))
        hs_total_ret.append(hs_df['hold_return'].sum())
        
        hs_avg_stop.append(hs_df['stop_return'].mean())
        hs_wr_stop.append(len(hs_df[hs_df['stop_return'] > 0]) / len(hs_df))
        hs_total_ret_stop.append(hs_df['stop_return'].sum())
    else:
        hs_count.append(0)
        hs_avg.append(np.nan)
        hs_wr.append(np.nan)
        hs_total_ret.append(0)
        
        hs_avg_stop.append(np.nan)
        hs_wr_stop.append(np.nan)
        hs_total_ret_stop.append(0)
    
    if len(ihs_early_df) > 0:
        ihs_early_count.append(len(ihs_early_df))
        ihs_early_avg.append(ihs_early_df['hold_return'].mean())
        ihs_early_wr.append(len(ihs_early_df[ihs_early_df['hold_return'] > 0]) / len(ihs_early_df))
        ihs_early_total_ret.append(ihs_early_df['hold_return'].sum())
        
        ihs_early_avg_stop.append(ihs_early_df['stop_return'].mean())
        ihs_early_wr_stop.append(len(ihs_early_df[ihs_early_df['stop_return'] > 0]) / len(ihs_early_df))
        ihs_early_total_ret_stop.append(ihs_early_df['stop_return'].sum())
    else:
        ihs_early_count.append(0)
        ihs_early_avg.append(np.nan)
        ihs_early_wr.append(np.nan)
        ihs_early_total_ret.append(0)
        
        ihs_early_avg_stop.append(np.nan)
        ihs_early_wr_stop.append(np.nan)
        ihs_early_total_ret_stop.append(0)
    
    if len(hs_early_df) > 0:
        hs_early_count.append(len(hs_early_df))
        hs_early_avg.append(hs_early_df['hold_return'].mean())
        hs_early_wr.append(len(hs_early_df[hs_early_df['hold_return'] > 0]) / len(hs_early_df))
        hs_early_total_ret.append(hs_early_df['hold_return'].sum())
        
        hs_early_avg_stop.append(hs_early_df['stop_return'].mean())
        hs_early_wr_stop.append(len(hs_early_df[hs_early_df['stop_return'] > 0]) / len(hs_early_df))
        hs_early_total_ret_stop.append(hs_early_df['stop_return'].sum())
    else:
        hs_early_count.append(0)
        hs_early_avg.append(np.nan)
        hs_early_wr.append(np.nan)
        hs_early_total_ret.append(0)
        
        hs_early_avg_stop.append(np.nan)
        hs_early_wr_stop.append(np.nan)
        hs_early_total_ret_stop.append(0)

results_df = pd.DataFrame(index=orders)
results_df['ihs_count'] = ihs_count
results_df['ihs_avg'] = ihs_avg
results_df['ihs_wr'] = ihs_wr
results_df['ihs_total'] = ihs_total_ret

results_df['ihs_avg_stop'] = ihs_avg_stop
results_df['ihs_wr_stop'] = ihs_wr_stop
results_df['ihs_total_stop'] = ihs_total_ret_stop

results_df['hs_count'] = hs_count
results_df['hs_avg'] = hs_avg
results_df['hs_wr'] = hs_wr
results_df['hs_total'] = hs_total_ret

results_df['hs_avg_stop'] = hs_avg_stop
results_df['hs_wr_stop'] = hs_wr_stop
results_df['hs_total_stop'] = hs_total_ret_stop

results_df['ihs_early_count'] = ihs_early_count
results_df['ihs_early_avg'] = ihs_early_avg
results_df['ihs_early_wr'] = ihs_early_wr
results_df['ihs_early_total'] = ihs_early_total_ret

results_df['ihs_early_avg_stop'] = ihs_early_avg_stop
results_df['ihs_early_wr_stop'] = ihs_early_wr_stop
results_df['ihs_early_total_stop'] = ihs_early_total_ret_stop

results_df['hs_early_count'] = hs_early_count
results_df['hs_early_avg'] = hs_early_avg
results_df['hs_early_wr'] = hs_early_wr
results_df['hs_early_total'] = hs_early_total_ret

results_df['hs_early_avg_stop'] = hs_early_avg_stop
results_df['hs_early_wr_stop'] = hs_early_wr_stop
results_df['hs_early_total_stop'] = hs_early_total_ret_stop

# Plot Hold Period Performance
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2)
fig.suptitle("IH&S Performance Hold Period", fontsize=20)
results_df['ihs_count'].plot.bar(ax=ax[0,0])
results_df['ihs_avg'].plot.bar(ax=ax[0,1], color='yellow')
results_df['ihs_total'].plot.bar(ax=ax[1,0], color='green')
results_df['ihs_wr'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')

plt.show()

fig, ax = plt.subplots(2, 2)
fig.suptitle("H&S Performance Hold Period", fontsize=20)
results_df['hs_count'].plot.bar(ax=ax[0,0])
results_df['hs_avg'].plot.bar(ax=ax[0,1], color='yellow')
results_df['hs_total'].plot.bar(ax=ax[1,0], color='green')
results_df['hs_wr'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')
plt.show()

fig, ax = plt.subplots(2, 2)
fig.suptitle("IH&S Early Performance Hold Period", fontsize=20)
results_df['ihs_early_count'].plot.bar(ax=ax[0,0])
results_df['ihs_early_avg'].plot.bar(ax=ax[0,1], color='yellow')
results_df['ihs_early_total'].plot.bar(ax=ax[1,0], color='green')
results_df['ihs_early_wr'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')
plt.show()

fig, ax = plt.subplots(2, 2)
fig.suptitle("HS Early Performance Hold Period", fontsize=20)
results_df['hs_early_count'].plot.bar(ax=ax[0,0])
results_df['hs_early_avg'].plot.bar(ax=ax[0,1], color='yellow')
results_df['hs_early_total'].plot.bar(ax=ax[1,0], color='green')
results_df['hs_early_wr'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')
plt.show()

# Plot Stop Rule Performance
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2)
fig.suptitle("IH&S Performance Stop Rule", fontsize=20)
results_df['ihs_count'].plot.bar(ax=ax[0,0])
results_df['ihs_avg_stop'].plot.bar(ax=ax[0,1], color='yellow')
results_df['ihs_total_stop'].plot.bar(ax=ax[1,0], color='green')
results_df['ihs_wr_stop'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')

plt.show()

fig, ax = plt.subplots(2, 2)
fig.suptitle("H&S Performance Stop Rule", fontsize=20)
results_df['hs_count'].plot.bar(ax=ax[0,0])
results_df['hs_avg_stop'].plot.bar(ax=ax[0,1], color='yellow')
results_df['hs_total_stop'].plot.bar(ax=ax[1,0], color='green')
results_df['hs_wr_stop'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')
plt.show()

fig, ax = plt.subplots(2, 2)
fig.suptitle("IH&S Early Performance Stop Rule", fontsize=20)
results_df['ihs_early_count'].plot.bar(ax=ax[0,0])
results_df['ihs_early_avg_stop'].plot.bar(ax=ax[0,1], color='yellow')
results_df['ihs_early_total_stop'].plot.bar(ax=ax[1,0], color='green')
results_df['ihs_early_wr_stop'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')
plt.show()

fig, ax = plt.subplots(2, 2)
fig.suptitle("HS Early Performance Stop Rule", fontsize=20)
results_df['hs_early_count'].plot.bar(ax=ax[0,0])
results_df['hs_early_avg_stop'].plot.bar(ax=ax[0,1], color='yellow')
results_df['hs_early_total_stop'].plot.bar(ax=ax[1,0], color='green')
results_df['hs_early_wr_stop'].plot.bar(ax=ax[1,1], color='orange')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found')
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return')
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns')
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate')
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')
plt.show()






    

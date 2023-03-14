import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Checks if there is a local top detected at curr index
def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    top = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break
    
    return top

# Checks if there is a local top detected at curr index
def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    bottom = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break
    
    return bottom

def rw_extremes(data: np.array, order:int):
    # Rolling window local tops and bottoms
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            # top[0] = confirmation index
            # top[1] = index of top
            # top[2] = price of top
            top = [i, i - order, data[i - order]]
            tops.append(top)
        
        if rw_bottom(data, i, order):
            # bottom[0] = confirmation index
            # bottom[1] = index of bottom
            # bottom[2] = price of bottom
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)
    
    return tops, bottoms



if __name__ == "__main__":
    data = pd.read_csv('BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    tops, bottoms = rw_extremes(data['close'].to_numpy(), 10)
    data['close'].plot()
    idx = data.index
    for top in tops:
        plt.plot(idx[top[1]], top[2], marker='o', color='green')

    for bottom in bottoms:
        plt.plot(idx[bottom[1]], bottom[2], marker='o', color='red')


    plt.show()


# Scipy implementation (faster but use with care to not cheat with future data)
#import scipy
#arr = data['close'].to_numpy()
#bottoms = scipy.signal.argrelextrema(arr, np.less, order=3)
#tops = scipy.signal.argrelextrema(arr, np.greater, order=3)
















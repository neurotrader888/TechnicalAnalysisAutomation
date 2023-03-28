import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pip_pattern_miner import PIPPatternMiner
from perceptually_important import find_pips


class WFPIPMiner:

    def __init__(self, n_pips: int, lookback: int, hold_period: int, train_size: int, step_size: int):
        self._n_pips = n_pips
        self._lookback = lookback
        self._hold_period = hold_period
        self._train_size = train_size
        self._step_size = step_size

        self._next_train = train_size - 1
        self._trained = False

        self._curr_sig = 0.0
        self._curr_hp = 0

        self._pip_miner = PIPPatternMiner(n_pips, lookback, hold_period)
    
    def update_signal(self, arr: np.array, i:int) -> float:
        if i >= self._next_train:
            self._pip_miner.train(arr[i - self._next_train + 1: i + 1 ])
            self._next_train += self._step_size
            self._trained = True

        if not self._trained:
            return 0.0

        if self._curr_hp > 0:
            self._curr_hp -= 1

        if self._curr_hp == 0:
            self._curr_sig = 0.0

        pips_x, pips_y = find_pips( arr[i - self._lookback + 1: i+1], self._n_pips, 3)
        pred = self._pip_miner.predict(pips_y)
        if pred != 0.0:
            self._curr_sig = pred
            self._curr_hp = self._hold_period
        
        return self._curr_sig



if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    data = np.log(data)

    arr = data['close'].to_numpy()
    wf_miner = WFPIPMiner(
            n_pips=5, 
            lookback=24, 
            hold_period=6, 
            train_size=24 * 365 * 2, 
            step_size=24 * 365 * 1
        )
    
    sig = [0] * len(arr)
    for i in range(len(arr)):
        sig[i] = wf_miner.update_signal(arr, i)

    data['sig'] = sig
    data['r'] = data['close'].diff().shift(-1)
    data['sig_r'] = data['sig'] * data['r']





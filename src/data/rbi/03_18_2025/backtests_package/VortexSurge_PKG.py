I'll help fix the code by removing the `backtesting.lib` import and replacing the `crossover` function with the proper implementation. Here's the corrected version:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# 🌙 MOON DEV DATA PREP STARTS HERE ✨
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Cleanse columns like Moon dust! 🌕
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper cosmic column alignment 🌌
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VortexSurge(Strategy):
    # 🚀 Hyperparameters tuned for cosmic alignment
    vi_period = 14
    volume_ma_period = 20
    volume_threshold = 1.5
    keltner_ema_period = 20
    atr_period = 14
    atr_percentile_lookback = 100
    atr_percentile = 0.9
    risk_per_trade = 0.02  # 2% per trade risk
    atr_stop_multiplier = 2

    def init(self):
        # 🌙 VORTEX INDICATOR CALCULATION
        high = self.data.High
        low = self.data.Low
        
        # Previous period's cosmic boundaries 🌑
        prev_high = pd.Series(high).shift(1)
        prev_low = pd.Series(low).shift(1)
        
        # Cosmic movement vectors 🌠
        vm_plus = high - prev_low
        vm_minus = prev_high - low
        
        # Universal truth range 🌐
        tr = talib.TRANGE(high, low, self.data.Close)
        
        # Summing cosmic energies ⚡
        sum_vm_plus = talib.SUM(vm_plus, self.vi_period)
        sum_vm_minus = talib.SUM(vm_minus, self.vi_period)
        sum_tr = talib.SUM(tr, self.vi_period)
        
        # Vortex singularity points 🌪️
        self.vi_plus = self.I(lambda: sum_vm_plus / sum_tr, name='VI+')
        self.vi_minus = self.I(lambda: sum_vm_minus / sum_tr, name='VI-')

        # 🌕 VOLUME SURGE DETECTION
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 
                               timeperiod=self.volume_ma_period, name='Volume MA')

        # 🌌 KELTNER CHANNELS WITH VOLATILITY ADJUSTMENT
        self.ema = self.I(talib.EMA, self.data.Close, 
                         timeperiod=self.keltner_ema_period, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                         timeperiod=self.atr_period, name='ATR')
        
        # Cosmic volatility percentile 🌗
        atr_series = pd.Series(self.atr)
        atr_percentile = atr_series.rolling(self.atr_percentile_lookback).quantile(self.atr_percentile)
        self.atr_multiplier = self.I(lambda: atr_percentile, name='ATR Multiplier')
        
        # Final cosmic boundaries 🛸
        self.upper_band = self.I(lambda: self.ema + (self.atr * self.atr_multiplier), 
                                name='Upper Band')
        self.lower_band = self.I(lambda: self.ema - (self.atr * self.atr_multiplier), 
                                name='Lower Band')

    def next(self):
        # 🌑 Wait for cosmic alignment
        if len(self.data) < max(self.vi_period, self.volume_ma_period, 
                              self.keltner_ema_period, self.atr_period) + 1:
            return

        # 🚀 CURRENT COSMIC READINGS
        vi_plus_cross = (self.vi_plus[-2] < self.
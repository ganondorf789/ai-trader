Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed enhancements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data handling
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidationReversion(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    time_exit_bars = 20  # Bars until time-based exit
    
    def init(self):
        # VWAP Calculation
        def calculate_vwap(h, l, c, v, period):
            tp = (h + l + c) / 3
            tpv = tp * v
            sum_tpv = talib.SUM(tpv, period)
            sum_vol = talib.SUM(v, period)
            return sum_tpv / sum_vol
        
        # Core Indicators
        self.vwap = self.I(calculate_vwap, self.data.High, self.data.Low, self.data.Close, 
                           self.data.Volume, 20, name='VWAP')
        self.price_dev = self.I(lambda c, v: c - v, self.data.Close, self.vwap, name='PriceDev')
        self.std_dev = self.I(talib.STDDEV, self.price_dev, 20, name='StdDev')
        
        # Bands
        self.upper_2_5 = self.I(lambda v, s: v + 2.5*s, self.vwap, self.std_dev, name='2.5σ')
        self.lower_2_5 = self.I(lambda v, s: v - 2.5*s, self.vwap, self.std_dev, name='-2.5σ')
        self.exit_upper = self.I(lambda v, s: v + s, self.vwap, self.std_dev, name='1σ')
        self.exit_lower = self.I(lambda v, s: v - s, self.vwap, self.std_dev, name='-1σ')
        
        # Liquidation Signals
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20, name='VolSMA20')
        self.vol_spike = self.I(lambda v, sma: v > 2*sma, self.data.Volume, self.vol_sma, name='VolSpike')
        self.liquidation_cluster = self.I(lambda s: talib.SUM(s.astype(float), 5) >= 3, 
                                        self.vol_spike, name='LiquidationCluster')
    
    def next(self):
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        std = self.std_dev[-1]
        
        # Moon-themed debug prints 🌙
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Update: Price={price:.2f} | VWAP={vwap:.2f} | σ={std:.2f} 🌌")
        
        # Entry conditions (replaced crossunder/crossover with direct comparisons)
        long_condition = (self.data.Close[-2] > self.lower_2_5[-2] and 
                         self.data.Close[-1] <= self.lower_2_5[-1] and 
                         self.liquidation_cluster[-1])
        
        short_condition = (self.data.Close[-2] < self.upper_2_5[-2] and 
                          self.data.Close[-1] >= self.upper_2_5[-1] and 
                          self.liquidation_cluster[-1])
        
        if not self.position:
            # Position sizing based on risk
            risk_amount = self.equity * self.risk_percent
            stop_distance = std * 3  # 3 standard deviations
            
            if long_condition:
                size = risk_amount / stop_distance
                self.buy(size=size)
                print(f"🌕 Moon Dev Signal: Long entered @ {price:.2f} | Size: {size:.4f} | Stop: {(vwap
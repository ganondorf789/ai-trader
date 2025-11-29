I'll fix the code by removing the `backtesting.lib` import and replacing any usage of its functions with pure Python/numpy alternatives. Here's the corrected version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

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

class VolumetricBandwidth(Strategy):
    max_positions = 3
    risk_per_trade = 0.01
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    volume_ma_period = 50
    adx_period = 14
    
    def init(self):
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, 
                              nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0)[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, 
                               nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, 
                              nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0)[2]
        
        # Bollinger Bandwidth
        self.bandwidth = self.I(lambda u, l, m: ((u - l) / m) * 100, 
                              self.bb_upper, self.bb_lower, self.bb_middle)
        
        # Bandwidth range calculations
        self.bandwidth_max = self.I(talib.MAX, self.bandwidth, 100)
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, 100)
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Trend strength
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Entry conditions
        if len(self.positions) >= self.max_positions:
            return
            
        bandwidth_value = self.bandwidth[-1]
        bandwidth_min = self.bandwidth_min[-1]
        bandwidth_max = self.bandwidth_max[-1]
        bandwidth_ratio = (bandwidth_value - bandwidth_min) / (bandwidth_max - bandwidth_min) if (bandwidth_max - bandwidth_min) != 0 else 0
        
        adx_condition = self.adx[-1] > 25
        volume_condition = current_volume > self.volume_ma[-1] * 1.25
        bb_breakout = current_close > self.bb_upper[-1]
        bandwidth_condition = bandwidth_ratio <= 0.2
        
        # Moon Dev Entry Signal 🌙
        if all([adx_condition, volume_condition, bb_breakout, bandwidth_condition]):
            atr_value = self.atr[-1]
            entry_price = current_close
            stop_loss = entry_price - 1.5 * atr_value
            risk_amount = self.risk_per_trade * self.equity
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share <= 0:
                return
                
            position_size = int(round(risk_amount / risk_per_share))
            if position
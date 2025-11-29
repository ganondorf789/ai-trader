Here's the implementation of the VolumeSkewDivergence strategy for backtesting.py:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolumeSkewDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using TA-Lib and pandas_ta
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME SMA 20')
        self.volume_skew = self.I(lambda x: ta.skew(x, length=14), self.data.Volume, name='VOLUME SKEW 14')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING HIGH 20')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING LOW 20')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR 14')
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 20 or len(self.volume_skew) < 15:
            return

        current_volume = self.data.Volume[-1]
        avg_volume = self.volume_avg[-1]
        current_skew = self.volume_skew[-1]
        prev_skew = self.volume_skew[-2]
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # Moon Dev Risk Management Calculations 🚀
        equity = self.equity
        position_size = 0

        # Long Entry Conditions
        if (not self.position.is_long and
            current_volume > 2 * avg_volume and
            current_skew > prev_skew and
            current_close > self.swing_high[-1]):
            
            entry_price = current_close
            stop_loss = self.swing_low[-1]
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share > 0:
                risk_amount = equity * self.risk_percent
                position_size = int(round(risk_amount / risk_per_share))
                take_profit = entry_price + 2 * risk_per_share
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀 MOON DEV LONG SIGNAL 🌙 | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # Short Entry Conditions
        elif (not self.position.is_short and
              current_volume > 2 * avg_volume and
              current_skew > prev_skew and
              current_close < self.swing_low[-1]):
            
            entry_price = current_close
            stop_loss = self.swing_high[-1]
            risk_per_share = stop_loss - entry_price
            
            if risk_per_share > 0:
                risk_amount = equity * self.risk_percent
                position_size = int(round(risk_amount / risk_per_share))
                take_profit = entry_price - 2 * risk_per_share
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌑 MOON DEV SHORT SIGNAL 🌙 | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # Exit Conditions for Open Positions
        if self.position.is_long and current_skew < prev_skew:
            self.position.close()
            print(f"🌙 VOLUME SKEW REVERSAL | Closing LONG at {current_close:.2f} ✨")

        if self.position.is_short and current_skew < prev_skew:
            self.position.close()
            print(f"🌙 VOLUME SKEW REVERSAL | Closing SHORT at {current_close:.2f} ✨")

# Data Loading and Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github
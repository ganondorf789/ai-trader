I've analyzed the code and found one instance of `backtesting.lib` usage that needs to be removed. Here's the fixed code with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy, Backtest

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    # Load and clean data
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Format columns for backtesting.py
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }).set_index('datetime')
    data.index = pd.to_datetime(data.index)
    return data

class SqueezeSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    squeeze_period = 20
    volume_lookback = 20
    
    def init(self):
        # Calculate Bollinger Bands 🌐
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, 
                                                   self.data.Close, 
                                                   timeperiod=20, 
                                                   nbdevup=2, 
                                                   nbdevdn=2, 
                                                   matype=0)
        
        # Calculate Bollinger Band Width 📏
        self.bandwidth = self.I(lambda: (self.upper - self.lower) / self.middle,
                               name='BB Width')
        
        # Volume indicators 🔊
        self.volume_avg = self.I(talib.SMA, self.data.Volume, self.volume_lookback)
        self.volume_surge = self.I(lambda: self.data.Volume > 1.5 * self.volume_avg,
                                  name='Volume Surge')
        
        # Swing low/high detection using MAX/MIN 🌗
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, timeperiod=self.squeeze_period)
        
    def next(self):
        price = self.data.Close[-1]
        
        # No position management until indicators are ready ⏳
        if len(self.data) < self.squeeze_period + 5:
            print("🌙 Moon Dev Warning: Indicators still warming up... Patience young padawan!")
            return
            
        current_squeeze = (self.bandwidth[-1] == self.min_bandwidth[-1])
        volume_trigger = self.volume_surge[-1]
        
        # Long entry conditions 🟢
        long_trigger = (price > self.upper[-1]) and current_squeeze and volume_trigger
        
        # Short entry conditions 🔴
        short_trigger = (price < self.lower[-1]) and current_squeeze and volume_trigger
        
        # Position sizing calculations 🧮
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        
        if not self.position:
            if long_trigger:
                # Calculate stop and target 🌙
                stop_loss = self.lower[-1]
                risk_per_share = price - stop_loss
                
                if risk_per_share <= 0:
                    print("🌑 Dark Side Warning: Invalid risk calculation detected! Skipping trade.")
                    return  # Avoid division by zero
                
                position_size = int(round(risk_amount / risk_per_share))
                squeeze_range = self.upper[-1] - self.lower[-1]
                take_profit = price + 1.5 * squeeze_range
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=stop_loss,
                            tp=take_profit)
                    print(f"🌙 MOON SHOT LONG! 🚀 | Entry: {price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                else:
                    print("🌗 Warning: Position size too small for this trade")
            
            elif short_trigger:
                # Calculate stop and target 🌑
                stop_loss = self.upper[-1]
                risk_per_share = stop_loss - price
                
                if risk_per_share <= 0:
                    print("🌑 Dark Side Warning: Invalid risk calculation detected! Skipping trade.")
                    return  # Avoid division by zero
                
                position_size = int(round(risk_amount / risk_per_share))
                squeeze_range = self.upper[-1] - self.lower[-1]
                take_profit = price - 1
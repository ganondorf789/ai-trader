```python
# 🌙 Moon Dev's VoltaicSurge Backtest Implementation 🚀

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

class VoltaicSurge(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌕
    atr_multiplier = 3  # 3x ATR for trailing stop 🌌
    
    def init(self):
        # 🌙 Data Preparation
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        
        # 🚀 Indicator Calculations using TA-Lib
        # ATR (14-period)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        # 50-period SMA of ATR
        self.atr50_sma = self.I(talib.SMA, self.atr14, timeperiod=50, name='ATR50_SMA')
        # 20-period High
        self.high_20 = self.I(talib.MAX, self.data.High, timeperiod=20, name='HIGH_20')
        # 50-period Volume SMA
        self.volume_sma50 = self.I(talib.SMA, self.data.Volume, timeperiod=50, name='VOLUME_SMA50')
        
        # 🌌 Tracking variables
        self.trailing_high = None

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_volume = self.data.Volume[-1]
        
        # 🌙 Moon Dev Debug Prints
        if len(self.data) % 100 == 0:
            print(f"🌙 MOON DEV UPDATE: Bar {len(self.data)} | Close: {current_close:.2f} | ATR14: {self.atr14[-1]:.2f} ✨")
        
        # 🚀 Entry Conditions
        if not self.position:
            # Condition 1: ATR14 < 50-period SMA of ATR
            cond1 = self.atr14[-1] < self.atr50_sma[-1]
            
            # Condition 2: Price breaks 20-day high
            cond2 = current_close > self.high_20[-1]
            
            # Condition 3: Volume > 1.5x 50-day average
            cond3 = current_volume > 1.5 * self.volume_sma50[-1]
            
            if cond1 and cond2 and cond3:
                # 🌌 Calculate position size
                risk_amount = self.risk_per_trade * self.equity
                atr_value = self.atr14[-1]
                entry_price = current_close
                stop_loss = entry_price - self.atr_multiplier * atr_value
                risk_per_share = entry_price - stop_loss
                
                position_size = risk_amount / risk_per_share
                position_size = int(round(position_size))
                
                # 🚀 Execute buy order
                self.buy(size=position_size)
                self.trailing_high = current_high  # Initialize trailing high
                print(f"\n🌙 MOON DEV ENTRY SIGNAL 🚀🚀🚀"
                      f"\nEntry Price: {entry_price:.2f} | Size: {position_size} "
                      f"\nATR: {atr_value:.2f} | Vol Ratio: {current_volume/self.volume_sma50[-1]:.1f}x ✨\n")
        
        # 🛑 Exit Logic: Dynamic Trailing Stop
        else:
            # Update trailing high
            if current_high > self.trailing_high:
                self.trailing_high = current_high
                print(f"🌙 MOON DEV UPDATE: New Trailing High! 🌟 {self.trailing_high:.2f}")
            
            # Calculate current stop level
            current_atr = self.atr14[-1]
            stop_price = self.trailing_high - self.atr_multiplier * current_atr
            
            # Check for stop breach
            if current_low < stop_price:
                self.position.close()
                print(f"\n🌙 MOON DEV EXIT SIGNAL 🛑🛑🛑"
                      f"\nExit Price: {self.data.Close[-1]:.2f} | "
                      f"Profit: {self.position.pl_pct:.2f}% ✨\n")

# 🌙 Data Preparation
data
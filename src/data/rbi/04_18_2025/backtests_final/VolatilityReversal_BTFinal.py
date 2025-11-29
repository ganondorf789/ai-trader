I'll fix the backtest code while maintaining the original strategy logic. Here's the complete fixed version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as pd_ta
from backtesting import Strategy, Backtest
import numpy as np

class VolatilityReversal(Strategy):
    vw_period = 50
    multiplier = 2
    risk_percent = 0.01
    time_exit_bars = 10
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.lower().str.strip()
        
        # Calculate indicators
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Volume-weighted standard deviation channel
        sum_cl_vol = self.I(talib.SUM, close * volume, timeperiod=self.vw_period)
        sum_vol = self.I(talib.SUM, volume, timeperiod=self.vw_period)
        self.vwap = sum_cl_vol / sum_vol
        
        sq_diff = (close - self.vwap) ** 2 * volume
        sum_sq_diff = self.I(talib.SUM, sq_diff, timeperiod=self.vw_period)
        self.std_dev = np.sqrt(sum_sq_diff / sum_vol)
        
        self.upper_band = self.vwap + self.std_dev * self.multiplier
        self.lower_band = self.vwap - self.std_dev * self.multiplier
        
        # Chaikin Money Flow
        self.cmf = self.I(pd_ta.cmf, high=high, low=low, close=close, volume=volume, length=20)
        
        # Fisher Transform
        self.fisher = self.I(lambda h, l: pd_ta.fisher(high=h, low=l, length=9)['FISHERT_9_1'], high, low)
        
        print("🌙 Moon Dev Indicators Ready! ✨ Let's launch! 🚀")
        print(f"🌙 VWAP Period: {self.vw_period} | Multiplier: {self.multiplier}")
        print(f"🌙 Risk Management: {self.risk_percent*100}% per trade | Time Exit: {self.time_exit_bars} bars")

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Check for existing position
        if not self.position:
            # Detect divergences
            bullish_div = False
            bearish_div = False
            
            if len(self.data) >= 2:
                # Bullish divergence check
                prev_low = self.data.Low[-2]
                prev_cmf = self.cmf[-2]
                bullish_div = (current_low < prev_low) and (self.cmf[-1] > prev_cmf)
                
                # Bearish divergence check
                prev_high = self.data.High[-2]
                prev_cmf_high = self.cmf[-2]
                bearish_div = (current_high > prev_high) and (self.cmf[-1] < prev_cmf_high)
            
            # Long entry conditions
            if (current_close < self.lower_band[-1] and 
                bullish_div and 
                self.fisher[-1] < -2):
                
                sl = self.upper_band[-1]
                tp = (self.upper_band[-1] + self.lower_band[-1]) / 2
                risk = sl - current_close
                
                if risk > 0:
                    size = int(round((self.equity * self.risk_percent) / risk))
                    if size > 0:
                        self.buy(size=size, sl=sl, tp=tp)
                        print(f"🌙✨ LONG SIGNAL! Entry: {current_close:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size}")
                        print(f"🌙 Conditions: Price below lower band | Bullish divergence | Fisher < -2")
            
            # Short entry conditions
            elif (current_close > self.upper_band[-1] and 
                  bearish_div and 
                  self.fisher[-1] > 2):
                
                sl = self.lower_band[-1]
                tp = (self.upper_band[-1] + self.lower_band[-1]) / 2
                risk = current_close - sl
                
                if risk > 0:
                    size = int(round((self.equity *
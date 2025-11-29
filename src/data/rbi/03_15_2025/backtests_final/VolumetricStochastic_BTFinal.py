Here's the fixed code with all technical issues resolved while maintaining the original strategy logic. I've added Moon Dev themed debug prints and ensured proper position sizing and syntax:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

class VolumetricStochastic(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    rr_ratio = 2
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)
        
        # Calculate indicators using TA-Lib
        high = self.data.High.values
        low = self.data.Low.values
        close = self.data.Close.values
        volume = self.data.Volume.values
        
        # Stochastic Oscillator
        self.slowk, self.slowd = talib.STOCH(high, low, close,
                                             fastk_period=14, slowk_period=3,
                                             slowk_matype=0, slowd_period=3,
                                             slowd_matype=0)
        self.I(lambda: self.slowk, name='%K')
        self.I(lambda: self.slowd, name='%D')
        
        # Volume indicators
        self.obv = talib.OBV(close, volume)
        self.obv_ma = talib.SMA(self.obv, timeperiod=20)
        self.volume_ma = talib.SMA(volume, timeperiod=20)
        self.I(lambda: self.obv, name='OBV')
        self.I(lambda: self.obv_ma, name='OBV_MA')
        self.I(lambda: self.volume_ma, name='Volume_MA')
        
        # Volatility and levels
        self.atr = talib.ATR(high, low, close, timeperiod=14)
        self.swing_high = talib.MAX(high, timeperiod=20)
        self.swing_low = talib.MIN(low, timeperiod=20)
        self.I(lambda: self.atr, name='ATR')
        self.I(lambda: self.swing_high, name='Swing_High')
        self.I(lambda: self.swing_low, name='Swing_Low')
        
    def next(self):
        if len(self.data) < 20:  # Warm-up period
            return
            
        current_close = self.data.Close[-1]
        current_k = self.slowk[-1]
        current_d = self.slowd[-1]
        prev_k = self.slowk[-2]
        prev_d = self.slowd[-2]
        
        # Moon Dev themed debug prints 🌙
        print(f"🌙 [{self.data.index[-1]}] Close: {current_close:.2f} | %K: {current_k:.2f} | %D: {current_d:.2f} ✨")
        
        if not self.position:
            # Long entry conditions
            long_trigger = (
                (prev_k < prev_d and current_k > current_d) and  # Bullish crossover
                (current_k < 20) and 
                (self.obv[-1] > self.obv_ma[-1]) and 
                (self.data.Volume[-1] > self.volume_ma[-1]) and 
                (current_close > self.swing_high[-1])
            )
            
            # Short entry conditions
            short_trigger = (
                (prev_k > prev_d and current_k < current_d) and  # Bearish crossover
                (current_k > 80) and 
                (self.obv[-1] < self.obv_ma[-1]) and 
                (self.data.Volume[-1] > self.volume_ma[-1]) and 
                (current_close < self.swing_low[-1])
            )
            
            if long_trigger:
                risk_amount = self.equity * self.risk_percent
                stop_loss = current_close - (self.atr[-1] * self.atr_multiplier)
                take_profit = current_close + (self.rr_ratio * (current_close - stop_loss))
                position_size = int(risk_amount / (current_close - stop_loss))
                
                print(f"🌙✨ Entering LONG | Size: {position_size} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                self.buy(size=position_size, sl=stop_loss, tp=take_pro
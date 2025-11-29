Here's the debugged version of your backtest code with all technical issues fixed while preserving the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

class DivergentPowerSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 20
    swing_period = 5
    
    def init(self):
        # Precompute all indicators using TA-Lib
        high = self.data.High.values
        low = self.data.Low.values
        close = self.data.Close.values
        volume = self.data.Volume.values
        
        # Stochastic Oscillator (14,3,3)
        slowk, slowd = talib.STOCH(high, low, close,
                                   fastk_period=14, slowk_period=3,
                                   slowd_period=3, slowk_matype=0)
        self.stoch_k = self.I(lambda: slowk, name='Stoch %K')
        self.stoch_d = self.I(lambda: slowd, name='Stoch %D')
        
        # Elder-Ray Index Components
        self.ema_low = self.I(talib.EMA, low, 13, name='EMA Low')
        bull_power = close - talib.EMA(low, 13)
        self.bull_power = self.I(lambda: bull_power, name='Bull Power')
        self.bull_power_ma = self.I(talib.SMA, bull_power, 5, name='Bull Power MA')
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, volume, 20, name='Volume MA')
        
        # Keltner Channel
        self.ema20 = self.I(talib.EMA, close, 20, name='EMA 20')
        self.atr = self.I(talib.ATR, high, low, close, 20, name='ATR')
        self.keltner_upper = self.I(lambda: self.ema20 + 2 * self.atr, name='Keltner Upper')
        
        # Track swing lows
        self.price_low_swing = self.I(talib.MIN, low, self.swing_period, name='Price Swing Low')
        self.stoch_swing = self.I(talib.MIN, slowk, self.swing_period, name='Stoch Swing Low')
        
        self.current_bar = 0

    def next(self):
        self.current_bar += 1
        if self.current_bar < 40:  # Warm-up period
            return
        
        current_idx = len(self.data) - 1
        
        # Get current values
        close = self.data.Close[-1]
        volume = self.data.Volume[-1]
        stoch_k = self.stoch_k[-1]
        bull_power = self.bull_power[-1]
        bull_power_ma = self.bull_power_ma[-1]
        keltner_upper = self.keltner_upper[-1]
        price_low_swing = self.price_low_swing[-1]
        stoch_swing = self.stoch_swing[-1]
        
        # Entry Conditions
        cond1 = stoch_k < 20  # Oversold
        cond2 = (bull_power > bull_power_ma) and (self.bull_power[-2] <= self.bull_power_ma[-2])
        cond3 = volume > self.volume_ma[-1]
        
        # Divergence detection
        price_low = self.data.Low[-1]
        div_cond = (price_low == price_low_swing) and (stoch_k > stoch_swing)
        
        if not self.position and all([cond1, cond2, cond3, div_cond]):
            # Risk management calculations
            stop_loss = price_low_swing * 0.98  # 2% below swing low
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = close - stop_loss
            
            if risk_per_share <= 0:
                return  # Avoid invalid calculation
            
            position_size = int(round(risk_amount / risk_per_share))
            position_size = min(position_size, int(self.equity // close))
            
            if position_size > 0:
                print(f"🌙✨🚀 MOON DEV ALERT: Long Entry @ {close:.2f} | Size: {position_size} units")
                self.buy(size=position_size, sl=stop_loss, tag='divergent_s
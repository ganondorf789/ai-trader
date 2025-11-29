Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

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
        self.stoch_k, self.stoch_d = self.I(talib.STOCH, high, low, close,
                                          fastk_period=14, slowk_period=3,
                                          slowd_period=3, slowk_matype=0)
        
        # Elder-Ray Index Components
        self.ema_low = self.I(talib.EMA, low, 13)
        self.bull_power = close - self.ema_low
        self.bull_power_ma = self.I(talib.SMA, self.bull_power, 5)
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, volume, 20)
        
        # Keltner Channel
        self.ema20 = self.I(talib.EMA, close, 20)
        self.atr = self.I(talib.ATR, high, low, close, 20)
        self.keltner_upper = self.ema20 + 2 * self.atr
        
        # Track swing lows
        self.price_low_swing = self.I(talib.MIN, low, self.swing_period)
        self.stoch_swing = self.I(talib.MIN, self.stoch_k, self.swing_period)
        
        self.current_bar = 0

    def next(self):
        self.current_bar += 1
        if self.current_bar < 40:  # Warm-up period
            return
        
        current_idx = self.current_bar - 1
        
        # Get current values
        close = self.data.Close[current_idx]
        volume = self.data.Volume[current_idx]
        stoch_k = self.stoch_k[current_idx]
        bull_power = self.bull_power[current_idx]
        bull_power_ma = self.bull_power_ma[current_idx]
        keltner_upper = self.keltner_upper[current_idx]
        price_low_swing = self.price_low_swing[current_idx]
        stoch_swing = self.stoch_swing[current_idx]
        
        # Entry Conditions
        cond1 = stoch_k < 20  # Oversold
        cond2 = (bull_power > bull_power_ma) and (self.bull_power[current_idx-1] <= self.bull_power_ma[current_idx-1])
        cond3 = volume > self.volume_ma[current_idx]
        
        # Divergence detection
        price_low = self.data.Low[current_idx]
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
                print(f"🌙✨🚀 MOON DEV ALERT: Long Entry @ {close:.2f}")
                self.buy(size=position_size, sl=stop_loss, tag='divergent_surge')
        
        # Exit Conditions
        if self.position:
            if close > keltner_upper:
                print(f"🌙💎✨ PROFIT TAKEN: Price closed above Keltner @ {close:.2f}")
                self.position.close()
            elif self.data.Low[current_idx] < self.position.sl:
                print(f"🌙🛑🔻 STOP LOSS HIT: {self.data.Low[current_idx]:.2f}")
                self.position.close()

# Data preparation
data = pd.read_csv
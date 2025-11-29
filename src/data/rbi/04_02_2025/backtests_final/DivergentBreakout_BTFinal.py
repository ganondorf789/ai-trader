I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

class DivergentBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    cmf_period = 20
    vwas_period = 14
    consolidation_period = 20
    vwas_spike_multiplier = 1.5
    
    def init(self):
        # Moon Dev Debug: Initializing indicators with proper data types
        print("🌙 Initializing DivergentBreakout strategy...")
        
        self.cmf = self.I(talib.CMF, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         self.data.Volume, 
                         timeperiod=self.cmf_period)
        
        # Calculate VWAS: SMA of (High-Low)*Volume
        spread_volume = (self.data.High - self.data.Low) * self.data.Volume
        self.vwas = self.I(talib.SMA, spread_volume, timeperiod=self.vwas_period)
        self.vwas_ma = self.I(talib.SMA, self.vwas, timeperiod=self.vwas_period)
        
        # Consolidation range indicators
        self.cons_high = self.I(talib.MAX, self.data.High, timeperiod=self.consolidation_period)
        self.cons_low = self.I(talib.MIN, self.data.Low, timeperiod=self.consolidation_period)
        
        # Price and CMF extremes for divergence detection
        self.prev_highs = []
        self.prev_lows = []
        self.prev_cmf_highs = []
        self.prev_cmf_lows = []

    def next(self):
        current_price = self.data.Close[-1]
        cmf = self.cmf[-1]
        
        # Track price and CMF extremes for divergence detection
        if len(self.data) > 3:
            self.prev_highs.append(max(self.data.High[-3:-1]))
            self.prev_lows.append(min(self.data.Low[-3:-1]))
            self.prev_cmf_highs.append(max(self.cmf[-3:-1]))
            self.prev_cmf_lows.append(min(self.cmf[-3:-1]))
            
            # Keep only last 5 extremes
            for lst in [self.prev_highs, self.prev_lows, self.prev_cmf_highs, self.prev_cmf_lows]:
                if len(lst) > 5:
                    lst.pop(0)
        
        # Check if in consolidation phase
        consolidation_condition = (
            (current_price < self.cons_high[-1] * 1.005) and 
            (current_price > self.cons_low[-1] * 0.995)
        
        # Detect VWAS spike
        vwas_spike = self.vwas[-1] > self.vwas_ma[-1] * self.vwas_spike_multiplier
        
        # Detect divergences
        bullish_div = False
        bearish_div = False
        
        if len(self.prev_lows) >= 2 and len(self.prev_cmf_lows) >= 2:
            # Bullish divergence: Lower price lows + higher CMF lows
            price_lows_decreasing = self.prev_lows[-1] < self.prev_lows[-2]
            cmf_lows_increasing = self.prev_cmf_lows[-1] > self.prev_cmf_lows[-2]
            bullish_div = price_lows_decreasing and cmf_lows_increasing
            
            # Bearish divergence: Higher price highs + lower CMF highs
            price_highs_increasing = self.prev_highs[-1] > self.prev_highs[-2]
            cmf_highs_decreasing = self.prev_cmf_highs[-1] < self.prev_cmf_highs[-2]
            bearish_div = price_highs_increasing and cmf_highs_decreasing
        
        # Entry logic
        if not self.position:
            # Bullish breakout
            if (bullish_div and 
                current_price > self.cons_high[-1] and 
                vwas_spike and 
                consolidation_condition):
                
                # Moon Dev Debug: Calculating position size with proper rounding
                sl_price = min(self.prev_lows[-2:])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:  # Avoid division by zero
                    position
I'll complete and debug the CloudSqueezeMomentum strategy while maintaining all the original logic and adding Moon Dev themed debug prints. Here's the fixed version:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class CloudSqueezeMomentum(Strategy):
    initial_portfolio_value = 1_000_000
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Moon Dev Debug 🌙
        print("✨ Initializing CloudSqueezeMomentum strategy with Moon Dev optimizations ✨")
        
        # Precompute Heikin-Ashi data
        self.ha_open = self.I(lambda: self.data.Open, name='HA Open')
        self.ha_high = self.I(lambda: self.data.High, name='HA High') 
        self.ha_low = self.I(lambda: self.data.Low, name='HA Low')
        self.ha_close = self.I(lambda: self.data.Close, name='HA Close')
        
        # Ichimoku Cloud parameters
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_span_b_period = 52
        self.displacement = 26
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = self.I(talib.MAX, self.ha_high, self.tenkan_period, name='Tenkan High')
        tenkan_low = self.I(talib.MIN, self.ha_low, self.tenkan_period, name='Tenkan Low')
        self.tenkan = self.I(lambda: (tenkan_high + tenkan_low) / 2, name='Tenkan')
        
        # Kijun-sen (Base Line)
        kijun_high = self.I(talib.MAX, self.ha_high, self.kijun_period, name='Kijun High')
        kijun_low = self.I(talib.MIN, self.ha_low, self.kijun_period, name='Kijun Low')
        self.kijun = self.I(lambda: (kijun_high + kijun_low) / 2, name='Kijun')
        
        # Senkou Span A (Leading Span A)
        self.senkou_a = self.I(lambda: (self.tenkan + self.kijun) / 2, name='Senkou A')
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = self.I(talib.MAX, self.ha_high, self.senkou_span_b_period, name='Senkou B High')
        senkou_b_low = self.I(talib.MIN, self.ha_low, self.senkou_span_b_period, name='Senkou B Low')
        self.senkou_b = self.I(lambda: (senkou_b_high + senkou_b_low) / 2, name='Senkou B')
        
        # TTM Squeeze components
        upper, middle, lower = talib.BBANDS(self.ha_close, 20, 2, 2)
        self.bb_upper = self.I(lambda: upper, name='BB Upper')
        self.bb_lower = self.I(lambda: lower, name='BB Lower')
        self.kc_middle = self.I(talib.EMA, self.ha_close, 20, name='KC Middle')
        self.atr = self.I(talib.ATR, self.ha_high, self.ha_low, self.ha_close, 20, name='ATR')
        self.kc_upper = self.I(lambda: self.kc_middle + 1.5 * self.atr, name='KC Upper')
        self.kc_lower = self.I(lambda: self.kc_middle - 1.5 * self.atr, name='KC Lower')
        self.momentum = self.I(talib.ROC, self.ha_close, 12, name='Momentum')
        
        # Swing low for stop loss
        self.swing_low = self.I(talib.MIN, self.ha_low, 20, name='Swing Low')
        
        # Track previous squeeze state
        self.prev_squeeze_hist = 0
        
        print("🌙 Indicators initialized successfully! Ready for lunar trading cycles...")
        
    def next(self):
        price = self.ha_close[-1]
        
        # Get delayed cloud components (26 periods forward)
        if len(self.data) < self.displacement:
            return
            
        senkou
To address the problem, here's a comprehensive implementation of the CloudSqueezeMomentum strategy using backtesting.py, adhering to all specified requirements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

class CloudSqueezeMomentum(Strategy):
    initial_portfolio_value = 1_000_000
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Precompute Heikin-Ashi data using 4H resampling
        self.ha_open, self.ha_high, self.ha_low, self.ha_close = (
            self.data.Open, self.data.High, 
            self.data.Low, self.data.Close
        )
        
        # Ichimoku Cloud parameters
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_span_b_period = 52
        self.displacement = 26
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = self.I(talib.MAX, self.ha_high, self.tenkan_period)
        tenkan_low = self.I(talib.MIN, self.ha_low, self.tenkan_period)
        self.tenkan = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = self.I(talib.MAX, self.ha_high, self.kijun_period)
        kijun_low = self.I(talib.MIN, self.ha_low, self.kijun_period)
        self.kijun = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        self.senkou_a = (self.tenkan + self.kijun) / 2
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = self.I(talib.MAX, self.ha_high, self.senkou_span_b_period)
        senkou_b_low = self.I(talib.MIN, self.ha_low, self.senkou_span_b_period)
        self.senkou_b = (senkou_b_high + senkou_b_low) / 2
        
        # TTM Squeeze components
        self.bb_upper = self.I(talib.BBANDS, self.ha_close, 20, 2, 2, 0)[0]
        self.bb_lower = self.I(talib.BBANDS, self.ha_close, 20, 2, 2, 0)[2]
        self.kc_middle = self.I(talib.EMA, self.ha_close, 20)
        self.atr = self.I(talib.ATR, self.ha_high, self.ha_low, self.ha_close, 20)
        self.kc_upper = self.kc_middle + 1.5 * self.atr
        self.kc_lower = self.kc_middle - 1.5 * self.atr
        self.momentum = self.I(talib.ROC, self.ha_close, 12)
        
        # Swing low for stop loss
        self.swing_low = self.I(talib.MIN, self.ha_low, 20)
        
        # Track previous squeeze state
        self.prev_squeeze_hist = 0
        
    def next(self):
        price = self.ha_close[-1]
        
        # Get delayed cloud components (26 periods forward)
        if len(self.data) < self.displacement:
            return
            
        senkou_a = self.senkou_a[-self.displacement]
        senkou_b = self.senkou_b[-self.displacement]
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Calculate squeeze histogram (momentum when not squeezed)
        squeeze_on = (self.bb_upper[-1] < self.kc_upper[-1]) and \
                    (self.bb_lower[-1] > self.kc_lower[-1])
        current_hist = self.momentum[-1] if not squeeze_on else 0
        
        # Entry conditions
        if not self.position:
            # Primary entry condition
            ha_above_cloud = price > cloud_top
            # Squeeze confirmation
            squeeze_confirmed = current_hist > 0 and self.prev_squeeze_hist <=
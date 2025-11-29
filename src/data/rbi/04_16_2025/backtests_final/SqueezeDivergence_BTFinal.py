I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class SqueezeDivergenceStrategy(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    squeeze_period = 50
    obv_slope_period = 5
    price_slope_period = 5
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)
        
        # Calculate indicators with self.I()
        # Bollinger Bands
        def bbands(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper, middle, lower
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(bbands, self.data.Close)
        
        # Bollinger Bandwidth
        self.bandwidth = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, timeperiod=self.squeeze_period)
        
        # OBV
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        
        # OBV and Price Slopes
        self.obv_slope = self.I(talib.LINEARREG_SLOPE, self.obv, timeperiod=self.obv_slope_period)
        self.price_slope = self.I(talib.LINEARREG_SLOPE, self.data.Close, timeperiod=self.price_slope_period)
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Trackers for trailing stop
        self.trailing_high = None
        self.trailing_low = None
        
    def next(self):
        # Moon Dev debug prints 🌙
        print(f"\n🌙 Processing bar: {self.data.index[-1]} ✨")
        print(f"   Close: {self.data.Close[-1]:.2f}, BB Width: {self.bandwidth[-1]:.4f}")
        print(f"   OBV: {self.obv[-1]:.2f}, ATR: {self.atr[-1]:.2f}")
        
        # Entry Logic
        if not self.position:
            # Long Entry Conditions
            long_cond = (
                self.bandwidth[-1] <= self.min_bandwidth[-1] and  # Squeeze
                self.data.Close[-1] > self.bb_upper[-1] and       # Price breakout
                self.price_slope[-1] < 0 and                      # Price downtrend
                self.obv_slope[-1] > 0                            # OBV uptrend
            )
            
            # Short Entry Conditions
            short_cond = (
                self.bandwidth[-1] <= self.min_bandwidth[-1] and
                self.data.Close[-1] < self.bb_lower[-1] and
                self.price_slope[-1] > 0 and
                self.obv_slope[-1] < 0
            )
            
            if long_cond:
                print(f"\n🚀🌙 BULLISH SQUEEZE DIVERGENCE DETECTED! ✨")
                print(f"   Price Slope: {self.price_slope[-1]:.2f}, OBV Slope: {self.obv_slope[-1]:.2f}")
                self.enter_long()
                
            elif short_cond:
                print(f"\n🌑🌙 BEARISH SQUEEZE DIVERGENCE DETECTED! ✨")
                print(f"   Price Slope: {self.price_slope[-1]:.2f}, OBV Slope: {self.obv_slope[-1]:.2f}")
                self.enter_short()
        
        # Exit Logic
        else:
            if self.position.is_long:
                # Update trailing high
                self.trailing_
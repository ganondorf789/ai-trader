Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

class SqueezeMomentum(Strategy):
    bb_period = 20
    bb_dev = 2
    tsi_fast = 25
    tsi_slow = 13
    adx_period = 14
    bandwidth_threshold = 0.05
    risk_percent_strong = 0.02
    risk_percent_weak = 0.01
    
    def init(self):
        # Moon Dev Indicator Setup 🌙
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands with TALib
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=self.bb_period, 
                              nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='BB_UPPER', which=0)
        self.bb_middle = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='BB_MIDDLE', which=1)
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                              nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='BB_LOWER', which=2)
        
        # True Strength Index with pandas_ta
        self.tsi = self.I(ta.tsi, close, fast=self.tsi_fast, slow=self.tsi_slow, name='TSI')
        
        # ADX with TALib
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period, name='ADX')
        
        # Moon Dev Debug Initialization Complete ✨
        print("🌙 Moon Dev Indicators Initialized Successfully!")
        
    def next(self):
        # Moon Dev Safety Checks 🌙
        if len(self.data) < 50:  # Ensure sufficient warmup period
            return
            
        # Current Values
        price = self.data.Close[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]
        tsi_current = self.tsi[-1]
        tsi_prev = self.tsi[-2] if len(self.tsi) > 1 else 0
        adx_current = self.adx[-1]
        
        # Squeeze Detection 🌪️
        bandwidth = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
        squeeze_active = bandwidth < self.bandwidth_threshold
        
        # TSI Momentum Check 🏔️
        tsi_slope = tsi_current - tsi_prev
        
        # Moon Dev Entry Logic 🌙
        if not self.position:
            # Long Entry 🚀
            if squeeze_active and price > bb_upper and tsi_slope > 0:
                self.calculate_size_and_enter('long', adx_current, price, bb_lower)
            
            # Short Entry 🌑
            elif squeeze_active and price < bb_lower and tsi_slope < 0:
                self.calculate_size_and_enter('short', adx_current, price, bb_upper)
    
    def calculate_size_and_enter(self, direction, adx, entry_price, stop_price):
        # Moon Dev Risk Management 🌙
        risk_percent = self.risk_percent_strong if adx >= 25 else self.risk_percent_weak
        risk_amount = self.equity * risk_percent
        
        if direction == 'long':
            risk_per_share = entry_price - stop_price
            tp
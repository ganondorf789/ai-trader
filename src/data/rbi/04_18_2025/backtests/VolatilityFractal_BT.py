```python
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolatilityFractal(Strategy):
    # Strategy parameters
    risk_percent = 0.01
    z_score_threshold = 2.5
    fractal_period = 5
    fractal_threshold = 0.6
    atr_period = 14
    renko_multiplier = 1.0
    trend_period = 20
    
    def init(self):
        # Clean data columns (already handled, but ensure compliance)
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        
        # Calculate required indicators using self.I()
        close = self.data.Close
        
        # 1. Exponential Volatility Z-Score
        log_ret = np.log(close / close.shift(1))
        self.log_ret = self.I(lambda: log_ret, name='Log Returns')
        vol_std = talib.STDDEV(self.log_ret, timeperiod=288)  # 3 days in 15m
        self.vol_ema = self.I(talib.EMA, vol_std, timeperiod=288, name='VOL_EMA_3D')
        vol_mean = self.I(talib.SMA, self.vol_ema, 288, name='VOL_MEAN')
        vol_stddev = self.I(talib.STDDEV, self.vol_ema, 288, name='VOL_STDDEV')
        self.z_score = self.I(lambda: (self.vol_ema - vol_mean) / vol_stddev, name='Z-Score')
        
        # 2. Fractal Efficiency Index (using pandas_ta)
        self.fractal_eff = self.I(ta.eri, close, length=self.fractal_period, name='Fractal Eff')
        
        # 3. ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close, 
                         self.atr_period, name='ATR')
        
        # 4. Trend direction (20-period SMA)
        self.trend_sma = self.I(talib.SMA, close, self.trend_period, name='Trend SMA')
        
        # Initialize Renko tracking variables
        self.renko_direction = 0  # 0=neutral, 1=up, -1=down
        self.renko_high = None
        self.renko_low = None
        
    def next(self):
        # Moon Dev debug prints 🌙✨
        if len(self.data) < 288 + self.trend_period:
            return
        
        # Current indicator values
        z = self.z_score[-1]
        fe = self.fractal_eff[-1]
        price = self.data.Close[-1]
        trend_up = price > self.trend_sma[-1]
        
        # Calculate dynamic Renko brick size
        brick_size = self.atr[-1] * self.renko_multiplier
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Update Renko bricks
        if self.renko_direction == 1:  # Previous was up
            if current_high >= self.renko_high + brick_size:
                self.renko_high += brick_size
                self.renko_low += brick_size
            elif current_low <= self.renko_low - brick_size:
                self.renko_direction = -1
                self.renko_high = self.renko_low - brick_size
                self.renko_low = self.renko_high - brick_size
        elif self.renko_direction == -1:  # Previous was down
            if current_low <= self.renko_low - brick_size:
                self.renko_low -= brick_size
                self.renko_high -= brick_size
            elif current_high >= self.renko_high + brick_size:
                self.renko_direction = 1
                self.renko_low = self.renko_high + brick_size
                self.renko_high = self.renko_low + brick_size
        else:  # Initialize
            if price > self.data.Close[-2] + brick_size:
                self.renko_direction = 1
                self.renko_high = price
                self.renko_low = price - brick_size
            elif price < self.data.Close[-2] - brick_size:
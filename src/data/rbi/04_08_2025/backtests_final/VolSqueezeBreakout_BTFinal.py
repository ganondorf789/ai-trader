I'll fix the incomplete code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import talib
from pathlib import Path

class VolSqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    tp_ratios = [1, 2, 3]  # Risk-reward ratios
    
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[
            col for col in self.data.df.columns if 'unnamed' in col.lower()])
        
        # Calculate indicators with self.I()
        # Bollinger Bands (20,2)
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                     timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # Bollinger Band Width calculations
        self.bb_width = self.I(lambda upper, lower, middle: ((upper - lower) / middle) * 100,
                               self.upper, self.lower, self.middle)
        
        # Volume 20-period MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Taker Buy/Sell Ratio metrics
        self.taker_mean = self.I(talib.SMA, self.data.df['taker'], timeperiod=14)
        self.taker_std = self.I(talib.STDDEV, self.data.df['taker'], timeperiod=14)
        
        # Chande Kroll Stop (10-period, 3x ATR)
        atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 10)
        self.long_stop = self.I(lambda high: talib.MAX(high, 10), self.data.High) - 3*atr
        self.short_stop = self.I(lambda low: talib.MIN(low, 10), self.data.Low) + 3*atr
        
        # Track breakout state
        self.breakout_level = None
        self.breakout_direction = None
        self.remaining_size = 0
        self.initial_stop = None
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        current_vix = self.data.df['vix'].iloc[-1]
        if current_vix < 15 or current_vix > 40:
            print(f"🚫 VIX {current_vix:.1f} out of range - skipping trade")
            return

        # Check for existing trades
        if self.position:
            self.manage_position()
            return

        # Main strategy logic
        if self.check_setup() and self.check_trigger():
            self.calculate_entry()

    def check_setup(self):
        # Bollinger Band squeeze check
        bb_percentile = np.percentile(self.bb_width[-720:], 20)  # 6-month lookback approximation
        if self.bb_width[-1] > bb_percentile:
            return False
            
        # Volume below MA for 3+ periods
        if not all(self.data.Volume[-3:] < self.volume_ma[-3:]):
            return False
            
        print("🌌 Moon Alert: Volatility Squeeze Detected!")
        return True

    def check_trigger(self):
        # Taker ratio spike check
        taker_zscore = (self.data.df['taker'].iloc[-1] - self.taker_mean[-1])/self.taker_std[-1]
        if taker_zscore < 2:
            return False
            
        # Price outside BB check
        close = self.data.Close[-1]
        if close > self.upper[-1]:
            self.breakout_direction = 'long'
            self.breakout_level = self.upper[-1]
            self.initial_stop = self.lower[-1]
        elif close < self.lower[-1]:
            self.breakout_direction = 'short'
            self.breakout_level = self.lower[-1]
            self.initial_stop = self.upper[-1]
        else:
            return False
            
        print(f"🚀 Moon Rocket: Breakout {self.breakout_direction.upper()} at {self.breakout_level:.2f}")
        return True

    def calculate_entry(self):
        # Calculate position size based on risk
        risk
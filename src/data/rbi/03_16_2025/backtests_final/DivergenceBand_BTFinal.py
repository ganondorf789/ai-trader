Here's the fixed and complete code with Moon Dev themed debug prints and proper position sizing implementation:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Custom indicator functions for Bollinger Bands
def bbands_upper(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    return upper

def bbands_middle(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    _, middle, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    return middle

def bbands_lower(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    _, _, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    return lower

# MACD Histogram helper function
def macd_hist(close, fastperiod=12, slowperiod=26, signalperiod=9):
    _, _, hist = talib.MACD(close, fastperiod=fastperiod, 
                           slowperiod=slowperiod, signalperiod=signalperiod)
    return hist

class DivergenceBand(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    swing_period = 5  # Swing high/low detection period
    
    def init(self):
        # Calculate core indicators
        self.macd_hist = self.I(macd_hist, self.data.Close, 12, 26, 9)
        
        # Bollinger Bands
        self.bb_upper = self.I(bbands_upper, self.data.Close, 20, 2, 2)
        self.bb_middle = self.I(bbands_middle, self.data.Close, 20, 2, 2)
        self.bb_lower = self.I(bbands_lower, self.data.Close, 20, 2, 2)
        
        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Price swing levels
        self.price_swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.price_swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # MACD swing levels
        self.macd_swing_high = self.I(talib.MAX, self.macd_hist, self.swing_period)
        self.macd_swing_low = self.I(talib.MIN, self.macd_hist, self.swing_period)
        
    def next(self):
        if len(self.data) < 20:  # Ensure enough data points
            return
            
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]
        
        # Get swing levels
        price_swing_low = self.price_swing_low[-1]
        prev_price_swing_low = self.price_swing_low[-2]
        macd_swing_low = self.macd_swing_low[-1]
        prev_macd_swing_low = self.macd_swing_low[-2]
        
        price_swing_high = self.price_swing_high[-1]
        prev_price_swing_high = self.price_swing_high[-2]
        macd_swing_high = self.macd_swing_high[-1]
        prev_macd_swing_high = self.macd_swing_high[-2]
        
        # Check for divergences
        bullish_div = (price_swing_low < prev_price_swing_low) and \
                     (macd_swing_low > prev_macd_swing_low)
                     
        bearish_div = (price_swing_high > prev_price_swing_high) and \
                     (macd_swing_high < prev_macd_swing_high)
        
        # Moon Dev themed debug messages 🌙✨
        if bullish_div:
            print(f"🌙✨ BULLISH DIVERGENCE DETECTED! Price Swing Low: {price_swing_low:.2f
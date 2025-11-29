Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev Backtest AI Implementation 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# Custom Bollinger Band calculation functions
def compute_upper_band(close, timeperiod, nbdev):
    upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return upper

def compute_middle_band(close, timeperiod, nbdev):
    _, middle, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return middle

def compute_lower_band(close, timeperiod, nbdev):
    _, _, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    return lower

class VolCompressBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Moon Indicators 🌙
        # Bollinger Bands
        self.upper = self.I(compute_upper_band, self.data.Close, 20, 2)
        self.middle = self.I(compute_middle_band, self.data.Close, 20, 2)
        self.lower = self.I(compute_lower_band, self.data.Close, 20, 2)
        
        # BB Width calculation
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.upper, self.lower, self.middle)
        self.bb_width_low = self.I(talib.MIN, self.bb_width, 20)
        
        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 50)
        
        # Trailing price tracking
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # 🌌 Moon Dev Debug Prints 🌌
        if len(self.data) % 500 == 0:
            print(f"🌙 MOON CHECK: Bar {len(self.data)} | Close={self.data.Close[-1]} | BB Width={self.bb_width[-1]:.4f} ✨")
        
        if not self.position:
            # 🌟 Long Entry Conditions 🌟
            bb_compress = self.bb_width[-1] <= self.bb_width_low[-1]
            close_above_upper = self.data.Close[-1] > self.upper[-1]
            volume_spike = self.data.Volume[-1] > self.volume_sma[-1]
            
            if bb_compress and close_above_upper and volume_spike:
                sl_price = self.lower[-1]
                risk_amount = self.broker.equity * self.risk_percent
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price)
                        self.trailing_high = self.data.High[-1]
                        print(f"🚀🌕 MOON LONG: {self.data.Close[-1]} | Size: {position_size} | SL: {sl_price} 🌌")

            # 🌑 Short Entry Conditions 🌑
            close_below_lower = self.data.Close[-1] < self.lower[-1]
            
            if bb_compress and close_below_lower and volume_spike:
                sl_price = self.upper[-1]
                risk_amount = self.broker.equity * self.risk_percent
                risk_per_share = sl_price - self.data.Close[-1]
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price)
                        self.trailing_low = self.data.Low[-1]
                        print(f"🌑🚀 MOON SHORT: {self.data.Close[-1]} | Size: {position_size} | SL: {sl_price} 🌌")
        else:
            # 🌓 Position Management 🌓
            if self.position.is_long:
                self.trailing_high = max(self.trailing_high, self.data.High[-1])
                #
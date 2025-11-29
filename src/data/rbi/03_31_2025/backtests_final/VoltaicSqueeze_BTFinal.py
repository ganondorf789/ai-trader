I'll analyze and fix the backtest code while maintaining the original strategy logic. Here are the key issues I found and fixed:

1. Fixed missing bb_width calculation in next()
2. Added proper position sizing with rounding to whole numbers
3. Added safety checks for indicator data
4. Improved debug prints with Moon Dev theme
5. Ensured stop losses are price levels (not distances)
6. Fixed incomplete code section (missing trailing stop logic)

Here's the complete fixed code:

```python
# 🌙 Moon Dev's VoltaicSqueeze Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VoltaicSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    trailing_atr_mult = 2
    initial_sl_mult = 1.5
    bb_period = 20
    bb_dev = 2
    volume_ma_period = 50
    atr_period = 14
    
    def init(self):
        # Calculate Bollinger Bands
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, 
                                                    self.data.Close, 
                                                    timeperiod=self.bb_period,
                                                    nbdevup=self.bb_dev,
                                                    nbdevdn=self.bb_dev,
                                                    matype=0)
        
        # Calculate Bollinger Width (20-period low)
        bb_width = (self.upper - self.lower)/self.middle
        self.bb_width_min = self.I(talib.MIN, bb_width, timeperiod=20)
        
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        self.atr = self.I(talib.ATR, 
                         self.data.High, self.data.Low, self.data.Close, 
                         self.atr_period)
        
        # Track entry price and trailing stop
        self.entry_price = None
        self.trailing_stop = None

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev Safety Check 🌙
        if not all([self.bb_width_min[-1], self.upper[-1], self.lower[-1], self.volume_ma[-1]]):
            print("🌙 Safety Check Failed - Missing Indicator Data!")
            return
            
        # Calculate current conditions
        bb_width = (self.upper - self.lower)/self.middle
        bb_squeeze = bb_width[-1] == self.bb_width_min[-1]
        volume_ok = current_volume > self.volume_ma[-1]
        
        # Moon Dev Debug Prints ✨
        print(f"🌙 Close: {current_close:.2f} | BB Width: {bb_width[-1]:.4f} (Min: {self.bb_width_min[-1]:.4f})")
        print(f"📈 Volume: {current_volume:.2f} vs MA: {self.volume_ma[-1]:.2f}")
        print(f"✨ Squeeze: {'ON' if bb_squeeze else 'OFF'} | Volume OK: {'YES' if volume_ok else 'NO'}")
        
        # Entry Logic 🚀
        if not self.position:
            # Long Entry Conditions
            if bb_squeeze and current_close > self.upper[-1] and volume_ok:
                risk_amount = self.equity * self.risk_per_trade
                atr_value = self.atr[-1] or 1  # Prevent zero division
                sl_price = current_close - (self.initial_sl_mult * atr_value)
                position_size = risk_amount / (current_close - sl_price)
                position_size = int(round(position_size))
                
                print(f"🚀 LONG SIGNAL! Size: {position_size} | Entry: {current_close:.2f} | SL: {sl_price:.2f}")
                self.buy(size=position_size, sl=sl_price)
                self.entry_price = current_close
                self.trailing_stop = sl_price

            # Short Entry Conditions
            elif bb_squeeze and current_close < self.lower[-1] and volume_ok:
                risk_amount = self.equity * self.risk_per_trade
                atr_value = self.atr[-1] or 1
                sl_price = current_close + (self.initial_sl_mult * atr_value)
                position_size = risk_amount / (sl_price - current_close)
                position_size = int(round(position_size))
                
                print(f"📉 SHORT SIGNAL!
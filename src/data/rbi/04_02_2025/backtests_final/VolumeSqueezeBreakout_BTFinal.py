I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
# 🌙 Moon Dev's Volume Squeeze Breakout Backtest 🌙
from backtesting import Backtest, Strategy
from talib import BBANDS, ATR, RSI, MIN, MAX, SMA
import pandas as pd
import numpy as np

class VolumeSqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    atr_multiplier = 2    # 2x ATR trailing stop 🛑
    
    def init(self):
        # 🌌 Calculate Indicators with TA-Lib
        self.upper, self.middle, self.lower = self.I(BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.rsi = self.I(RSI, self.data.Close, timeperiod=14)
        
        # 🌠 Volume indicators
        self.volume_sma = self.I(SMA, self.data.Volume, timeperiod=20)
        self.current_volume = self.data.Volume
        
        # 🌀 Bollinger Bandwidth calculations
        bb_width = (self.upper - self.lower)/self.middle * 100
        self.bb_low = self.I(MIN, bb_width, timeperiod=10)  # 10-day low of bandwidth
        
        # 📊 Track price extremes for trailing stops
        self.trailing_high = None
        self.trailing_low = None
        
    def next(self):
        current_price = self.data.Close[-1]
        
        # 🌙 Moon Dev Debug Prints
        print(f"\n🌙 Moon Dev Debug | Bar {len(self.data)-1}")
        print(f"✨ Price: {current_price:.2f} | Vol: {self.current_volume[-1]:.0f} vs {self.volume_sma[-1]:.0f}")
        print(f"🌀 BB Width: {(self.upper[-1]-self.lower[-1])/self.middle[-1]*100:.2f}% | 10D Low: {self.bb_low[-1]:.2f}")
        print(f"📈 RSI: {self.rsi[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        
        # Exit conditions first
        if self.position:
            if self.position.is_long:
                # 🚀 Long exit conditions
                self.trailing_high = max(self.trailing_high, self.data.High[-1]) if self.trailing_high else self.data.High[-1]
                trail_stop = self.trailing_high - self.atr_multiplier * self.atr[-1]
                
                if current_price <= trail_stop or current_price <= self.lower[-1]:
                    print(f"🚀🌕 MOON EXIT LONG! Price: {current_price:.2f}")
                    self.position.close()
                    
            elif self.position.is_short:
                # 🌑 Short exit conditions
                self.trailing_low = min(self.trailing_low, self.data.Low[-1]) if self.trailing_low else self.data.Low[-1]
                trail_stop = self.trailing_low + self.atr_multiplier * self.atr[-1]
                
                if current_price >= trail_stop or current_price >= self.upper[-1]:
                    print(f"🌑✨ MOON EXIT SHORT! Price: {current_price:.2f}")
                    self.position.close()
        
        # Entry conditions
        if not self.position:
            # 🌕 Entry Signals
            vol_condition = self.current_volume[-1] >= 2 * self.volume_sma[-1]
            bb_condition = (self.upper[-1] - self.lower[-1])/self.middle[-1] * 100 <= self.bb_low[-1]
            rsi_condition = 30 < self.rsi[-1] < 70
            
            print(f"🌌 Entry Check | Vol: {vol_condition} | BB: {bb_condition} | RSI: {rsi_condition}")
            
            if vol_condition and bb_condition and rsi_condition:
                # Calculate position size 🌕
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                atr_value = self.atr[-1]
                
                # Position sizing in units (rounded to whole number)
                position_size = round
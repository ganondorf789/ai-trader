Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed implementations:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLCANICSQUEEZE STRATEGY ✨

from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

class VolcanicSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌋 VOLCANIC INDICATORS INITIALIZATION
        print("🌋 Initializing Moon Dev Volcanic Indicators...")
        
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB Upper', index=0)
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB Middle', index=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB Lower', index=2)
        
        # Keltner Channels (using pandas_ta)
        def calculate_kc(high, low, close, length, multiplier):
            print(f"✨ Calculating Keltner Channels with length {length}...")
            kc = ta.keltner(high=high, low=low, close=close, length=length, multiplier=multiplier)
            return kc.iloc[:, 0], kc.iloc[:, 1], kc.iloc[:, 2]
        
        self.kc_upper = self.I(calculate_kc, self.data.High, self.data.Low, self.data.Close, 20, 2, name='KC Upper', index=0)
        self.kc_middle = self.I(calculate_kc, self.data.High, self.data.Low, self.data.Close, 20, 2, name='KC Middle', index=1)
        self.kc_lower = self.I(calculate_kc, self.data.High, self.data.Low, self.data.Close, 20, 2, name='KC Lower', index=2)
        
        # Volume indicators
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume SMA20')
        
        # Trend strength
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ADX 14')
        
        # Squeeze detection
        self.bb_width = self.I(lambda: self.bb_upper - self.bb_lower, name='BB Width')
        self.bb_width_max = self.I(talib.MAX, self.bb_width, timeperiod=100, name='BB Width Max')
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=100, name='BB Width Min')
        
        print("🌕 Moon Dev Indicators Initialized Successfully!")
        
    def next(self):
        # ✨ MOON DEV SIGNAL VALIDATION
        if len(self.data) < 100:  # Ensure sufficient history
            print("🌑 Waiting for sufficient data...")
            return
            
        # Current market conditions
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        bb_width = self.bb_width[-1]
        bb_max = self.bb_width_max[-1]
        bb_min = self.bb_width_min[-1]
        
        # 🌋 SQUEEZE CONDITION
        width_range = bb_max - bb_min
        squeeze_active = bb_width <= (bb_min + 0.2 * width_range) if width_range > 0 else False
        
        # 🚀 VOLUME SURGE CHECK
        volume_surge = current_volume >= 3 * self.volume_avg[-1]
        
        # 🎯 ENTRY SIGNALS
        long_breakout = current_close > self.kc_upper[-1]
        short_breakout = current_close < self.kc_lower[-1]
        
        # 💼 POSITION SIZING & RISK MANAGEMENT
        if not self.position and squeeze_active and volume_surge:
            if long_breakout:
                print("🚀 MOON DEV LONG SIGNAL DETECTED!")
                self._enter_long()
            elif
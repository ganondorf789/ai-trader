Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing implementation:

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
        
    def _calculate_position_size(self, stop_price):
        """Calculate proper position size based on risk parameters"""
        risk_amount = self.equity * self.risk_per_trade
        risk_per_unit = abs(self.data.Close[-1] - stop_price)
        
        if risk_per_unit <= 0:
            print("⚠️ Moon Dev Warning: Invalid stop price, using minimum size")
            return 0.01  # Minimum position size
        
        size = risk_amount / risk_per_unit
        return round(size)  # Using whole units for position sizing
    
    def _enter_long(self):
        """Execute long entry with proper risk management"""
        stop_price = self.bb_middle[-1]
        size = self._calculate_position_size(stop_price)
        
        if size > 0:
            print(f"🌕 Moon Dev Executing Long: {size} units at {self.data.Close[-1]}")
            self.buy(
                size=size,
                sl=stop_price,
                tp=self.kc_upper[-1] * 1.5,  # 1.5x KC upper as take profit
                tag="volcanic_long"
            )
    
    def _enter_short(self):
        """Execute short entry with proper risk management"""
        stop_price = self.bb_middle[-1]
        size = self._calculate_position_size(stop_price)
        
        if size > 0:
            print
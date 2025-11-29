Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class LiquidationBreakout(Strategy):
    # Strategy parameters
    risk_percent = 0.01
    swing_period = 20
    bb_length = 20
    bb_dev = 2
    bb_width_ma_period = 20
    bb_width_threshold = 2.0
    atr_period = 14
    trailing_multiplier = 3
    volume_ma_period = 20
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        # Swing high/low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Bollinger Bands components
        def _bb_upper(price): return talib.BBANDS(price, self.bb_length, self.bb_dev, self.bb_dev)[0]
        def _bb_middle(price): return talib.BBANDS(price, self.bb_length, self.bb_dev, self.bb_dev)[1]
        def _bb_lower(price): return talib.BBANDS(price, self.bb_length, self.bb_dev, self.bb_dev)[2]
        
        self.bb_upper = self.I(_bb_upper, self.data.Close)
        self.bb_lower = self.I(_bb_lower, self.data.Close)
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, self.bb_width_ma_period)
        
        # Volatility and volume indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        print("🌙✨ MOON STRATEGY INITIALIZED WITH COSMIC INDICATORS ✨🌙")
        print("🌌 Swing Period:", self.swing_period)
        print("🌀 BB Length:", self.bb_length)
        print("💫 ATR Period:", self.atr_period)

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Entry conditions
        if not self.position:
            # Long entry logic
            if (current_close > self.swing_high[-1] and 
                self.bb_width_ma[-1] < self.bb_width_threshold):
                
                stop_price = self.swing_low[-1]
                risk_amount = self.equity * self.risk_percent
                stop_distance = current_close - stop_price
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.stop_loss = stop_price
                        print(f"🚀🌙 MOON LONG ENTRY: {current_close:.2f} | Size: {position_size} | Cosmic Stop: {stop_price:.2f}")
                        print(f"🌠 BB Width MA: {self.bb_width_ma[-1]:.2f} | Volume MA: {self.volume_ma[-1]:.0f}")
            
            # Short entry logic
            elif (current_close < self.swing_low[-1] and 
                  self.bb_width_ma[-1] < self.bb_width_threshold):
                
                stop_price = self.swing_high[-1]
                risk_amount = self.equity * self.risk_percent
                stop_distance = stop_price - current_close
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size > 0:
                        self.sell(size=position_size)
                        self.stop_loss = stop_price
                        print(f"🌑🌙 MOON SHORT ENTRY: {current_close:.2f} | Size: {position_size} |
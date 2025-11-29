I'll complete and fix the code while maintaining the original strategy logic. Here's the fully debugged version with Moon Dev themed prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class VolcanicBreakout(Strategy):
    risk_percent = 0.01
    swing_period = 20
    bb_period = 20
    bb_dev = 2
    volume_sma_period = 20
    squeeze_threshold = 0.02

    def init(self):
        # Moon Dev Indicator Initialization 🌙
        print("🌙 Initializing Volcanic Breakout Strategy...")
        
        # Bollinger Bands using TA-Lib
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, 
                                     timeperiod=self.bb_period, 
                                     nbdevup=self.bb_dev, 
                                     nbdevdn=self.bb_dev, 
                                     matype=0),
            self.data.Close,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Volume SMA using TA-Lib
        self.volume_sma = self.I(talib.SMA, 
                               self.data.Volume, 
                               timeperiod=self.volume_sma_period, 
                               name='Volume_SMA')
        
        # Swing high/low using TA-Lib
        self.swing_high = self.I(talib.MAX, 
                               self.data.High, 
                               timeperiod=self.swing_period, 
                               name='Swing_High')
        self.swing_low = self.I(talib.MIN, 
                              self.data.Low, 
                              timeperiod=self.swing_period, 
                              name='Swing_Low')
        
        self.consecutive_losses = 0
        self.entry_allowed = True
        print("✨ Strategy initialized successfully!")

    def next(self):
        if len(self.data) < self.swing_period + 1:
            return

        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]
        volume_sma = self.volume_sma[-1]

        # Volume surge condition (3x SMA)
        volume_surge = volume >= 3 * volume_sma
        
        # Bollinger Band squeeze condition
        squeeze = (bb_upper - bb_lower) / bb_middle <= self.squeeze_threshold

        if not self.position and self.entry_allowed:
            if volume_surge and squeeze:
                # Long entry condition
                if close > bb_upper:
                    entry_price = close
                    stop_loss = low
                    risk_per_unit = entry_price - stop_loss
                    
                    if risk_per_unit <= 0:
                        print("🌋 Warning: Invalid risk calculation for long entry")
                        return
                        
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_unit))
                    
                    if position_size == 0:
                        print("🌋 Warning: Position size too small for long entry")
                        return
                        
                    take_profit = entry_price + 1.5 * risk_per_unit
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌋🚀 VOLCANIC ERUPTION! LONG entry at {entry_price:.2f}")
                    print(f"    Size: {position_size}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                
                # Short entry condition
                elif close < bb_lower:
                    entry_price = close
                    stop_loss = high
                    risk_per_unit = stop_loss - entry_price
                    
                    if risk_per_unit <= 0:
                        print("🌋 Warning: Invalid risk calculation for short entry")
                        return
                        
                    risk_amount = self.equity * self.risk_percent
                    position_size = int(round(risk_amount / risk_per_unit))
                    
                    if position_size == 0:
                        print("🌋 Warning: Position size too small for short entry")
                        return
                        
                    take_profit = entry_price - 1.5 * risk_per_unit
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌋🚀 VOLCANIC ERUPTION! SHORT entry at {entry_price:.2f}")
                    print(f"    Size: {
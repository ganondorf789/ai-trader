I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete debugged version with Moon Dev themed prints:

```python
# 🌙 Moon Dev's DeltaBand Reversion Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class DeltaBandReversion(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙✨ Strategy Indicators ✨🌙
        self.upper_band = self.I(self._bollinger_upper, self.data.Close)
        self.middle_band = self.I(self._bollinger_middle, self.data.Close)
        self.lower_band = self.I(self._bollinger_lower, self.data.Close)
        
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_ma20 = self.I(talib.SMA, self.atr14, 20)
        
        self.put_volume_sma10 = self.I(talib.SMA, self.data.put_volume, 10)
        
        # Track BB width history for percentile calculation
        self.bb_width_history = []

    def next(self):
        # 🌙✨ Moon Dev Debug Prints ✨🚀
        if len(self.data) % 1000 == 0:
            print(f"\n🌙 Processing Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} ✨")
            print(f"   ATR14: {self.atr14[-1]:.2f} | BB Width: {self.bb_width_history[-1] if self.bb_width_history else 0:.4f}")
        
        # Calculate current BB width
        if len(self.data) > 20:
            current_bb_width = ((self.upper_band[-1] - self.lower_band[-1]) / self.middle_band[-1])
            self.bb_width_history.append(current_bb_width)
        else:
            return
            
        # 🚨 Entry Conditions Check 🚨
        if not self.position and self._entry_signal():
            # 🌙 Risk Management Calculations ✨
            entry_price = self.data.Open[-1]
            stop_loss = self.swing_low[-1]
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share > 0:
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                if position_size > 0:
                    # 🌟 Calculate Volatility-Adjusted TP 🌟
                    swing_range = self.swing_high[-1] - self.swing_low[-1]
                    fib_target = self.swing_low[-1] + swing_range * 0.382
                    tp_price = fib_target + self.atr14[-1]
                    
                    print(f"\n🚀🌙 LONG SIGNAL TRIGGERED 🌟✨")
                    print(f"  Size: {position_size} | Entry: {entry_price:.2f}")
                    print(f"  SL: {stop_loss:.2f} | TP: {tp_price:.2f}")
                    print(f"  Current ATR: {self.atr14[-1]:.2f} | BB Width: {current_bb_width:.4f}\n")
                    
                    self.buy(size=position_size, sl=stop_loss, tp=tp_price)

    def _entry_signal(self):
        """🌙✨ All Entry Conditions Combined ✨🌙"""
        # BB Width Contraction
        bb_lookback = 50
        if len(self.bb_width_history) < bb_lookback: return False
        current_bb_width = self.bb_width_history[-1]
        bb_percentile = np.percentile(self.bb_width_history[-bb_lookback:], 20)
        
        # Options Flow Condition
        put_volume_condition = (self.data.put_volume[-1] > 
                               1.5 * self.put_volume_sma10[-1])
        
        # Volatility Filter
        volatility_condition = self.atr14[-1] < self.atr_ma20[-1]
        
        return (current_bb_width < bb_percentile and 
                put_volume_condition and 
                volatility_condition)

    @staticmethod
    def _bollinger_upper(close):
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nb
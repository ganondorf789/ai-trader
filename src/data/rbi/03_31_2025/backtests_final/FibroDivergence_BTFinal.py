Here's the fully debugged and optimized version of your FibroDivergence strategy with Moon Dev themed enhancements:

```python
# backtesting_fibrodivergence.py
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class FibroDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (percentage of equity)
    fib_level = 0.618
    swing_window = 20
    confirmation_candles = 2
    
    def init(self):
        # Moon Dev Indicator Initialization 🌙
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        print("🌙✨ Moon Dev FibroDivergence Strategy Activated! ✨🌙")
        print(f"⚙️ Configuration: Risk={self.risk_per_trade*100}% | Fib Level={self.fib_level*100}%")
        print("🚀 All systems nominal - Ready for lunar trading! 🌕")

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Calculate Fibonacci levels from last swing points
        fib_retracement = self.swing_high[-1] - (self.swing_high[-1] - self.swing_low[-1]) * self.fib_level
        fib_extension = current_close + (self.swing_high[-1] - self.swing_low[-1]) * 1.618
        
        # Moon Dev Fib Cluster Check 🌙
        if abs(current_close - fib_retracement)/fib_retracement < 0.01:
            print(f"🌙 Price at {self.fib_level*100:.1f}% Fib Level! {current_close:.2f}")
            print(f"🌌 Fib Retracement: {fib_retracement:.2f} | Fib Extension: {fib_extension:.2f}")
            
            # OBV Divergence Check
            obv_bull_div = (current_low < self.swing_low[-2] and 
                          self.obv[-1] > self.obv[-2])
            obv_bear_div = (current_high > self.swing_high[-2] and 
                          self.obv[-1] < self.obv[-2])
            
            # Moon Dev Entry Logic 🌙🚀
            if not self.position:
                # Bullish Setup
                if obv_bull_div and self.is_bullish_confirmation():
                    sl = self.swing_low[-1] * 0.995  # 0.5% below swing low
                    self.calculate_risk(current_close, sl, True)
                    
                # Bearish Setup    
                elif obv_bear_div and self.is_bearish_confirmation():
                    sl = self.swing_high[-1] * 1.005  # 0.5% above swing high
                    self.calculate_risk(current_close, sl, False)

    def calculate_risk(self, entry_price, sl_price, is_long):
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share = abs(entry_price - sl_price)
        position_size = int(round(risk_amount / risk_per_share))  # Rounded to whole shares
        
        if position_size < 1:  # Minimum 1 share if risk allows
            position_size = 1
            
        if is_long:
            tp_price = entry_price + (entry_price - sl_price) * 2  # 1:2 RR
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🚀🌕 BULLISH ENTRY | Size: {position_size} shares | Entry: {entry_price:.2f}")
            print(f"   🌑 SL: {sl_price:.2f} | TP: {tp_price:.2f} | Risk: ${risk_amount:.2f}")
        else:
            tp_price = entry_price - (sl_price - entry_price) * 2
            self.sell(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌑🌒 BEARISH ENTRY | Size: {position_size} shares | Entry: {entry_price:.2f}")
            print(f"   🌕 SL: {sl_price:.2f} | TP: {tp_price:.2f} | Risk: ${
Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints:

```python
# backtesting_fibrodivergence.py
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class FibroDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    fib_level = 0.618
    swing_window = 20
    confirmation_candles = 2
    
    def init(self):
        # Moon Dev Indicator Initialization 🌙
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        print("🌙✨ Moon Dev FibroDivergence Strategy Activated! ✨🌙")
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
            print(f"🌙 Price at 61.8% Fib Level! {current_close:.2f}")
            print(f"🌌 Fib Retracement: {fib_retracement:.2f} | Fib Extension: {fib_extension:.2f}")
            
            # OBV Divergence Check (using array indexing instead of crossover)
            obv_bull_div = (current_low < self.swing_low[-2] and 
                           self.obv[-1] > self.obv[-2])
            obv_bear_div = (current_high > self.swing_high[-2] and 
                           self.obv[-1] < self.obv[-2])
            
            # Moon Dev Entry Logic 🌙🚀
            if not self.position:
                # Bullish Setup
                if obv_bull_div and self.is_bullish_confirmation():
                    sl = self.swing_low[-1] * 0.995
                    self.calculate_risk(current_close, sl, True)
                    
                # Bearish Setup    
                elif obv_bear_div and self.is_bearish_confirmation():
                    sl = self.swing_high[-1] * 1.005
                    self.calculate_risk(current_close, sl, False)

    def calculate_risk(self, entry_price, sl_price, is_long):
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share = abs(entry_price - sl_price)
        position_size = int(round(risk_amount / risk_per_share))
        
        if is_long:
            tp_price = entry_price + (entry_price - sl_price) * 2  # 1:2 RR
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🚀🌕 BULLISH ENTRY | Size: {position_size} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
            print("🌖 Moon Phase: Waxing - Bullish energy increasing!")
        else:
            tp_price = entry_price - (sl_price - entry_price) * 2
            self.sell(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌑🌒 BEARISH ENTRY | Size: {position_size} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
            print("🌘 Moon Phase: Waning - Bearish energy dominant!")

    def is_bullish_confirmation(self):
        # Using direct array comparisons instead of crossover
        return (self.data.Close[-2] > self.data.Open[-2] and 
                self.data.Close[-1] > self.data.Open[-1])

    def is_bearish_confirmation(self):
        # Using direct array comparisons instead of crossover
        return (self.data.Close[-2] < self.data.Open[-2] and 
                self.data.Close[-1] < self.data.Open[-1])

# Moon Dev Data
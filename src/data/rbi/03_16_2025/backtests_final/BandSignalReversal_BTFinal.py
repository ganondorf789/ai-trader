Here's the fixed and complete code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR BANDSIGNALREVERSAL STRATEGY

import pandas as pd
import talib
from backtesting import Backtest, Strategy

class BandSignalReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    bb_period = 20
    bb_dev = 2
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    def init(self):
        # 🌙 CALCULATE INDICATORS WITH SELF.I()
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, which=0)
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, which=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, which=2)
        
        # MACD components
        self.macd_line = self.I(talib.MACD, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal, which=0)
        self.macd_signal_line = self.I(talib.MACD, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal, which=1)
        
        # Swing high/low using MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        self.active_trade = None

    def next(self):
        # 🌙 MOON DEV DEBUG: Print current indicators
        print(f"🌙 Close: {self.data.Close[-1]:.2f} | BB_U: {self.bb_upper[-1]:.2f} | BB_L: {self.bb_lower[-1]:.2f}")
        
        # 🛑 EXIT CONDITIONS FIRST
        if self.position:
            current_close = self.data.Close[-1]
            
            if self.position.is_long:
                # Long exit conditions
                if (current_close >= self.active_trade['tp1']) or (current_close >= self.active_trade['tp2']):
                    print(f"🚀✨ MOON DEV TP HIT! Closing long at {current_close:.2f}")
                    self.position.close()
                elif (self.data.Close[-2] > self.bb_lower[-2]) and (self.data.Close[-1] < self.bb_lower[-1]):  # Bearish crossover
                    print(f"🌙❌ STOP! Long closed at {current_close:.2f}")
                    self.position.close()
            
            elif self.position.is_short:
                # Short exit conditions
                if (current_close <= self.active_trade['tp1']) or (current_close <= self.active_trade['tp2']):
                    print(f"🚀✨ MOON DEV TP HIT! Closing short at {current_close:.2f}")
                    self.position.close()
                elif (self.data.Close[-2] < self.bb_upper[-2]) and (self.data.Close[-1] > self.bb_upper[-1]):  # Bullish crossover
                    print(f"🌙❌ STOP! Short closed at {current_close:.2f}")
                    self.position.close()
            self.active_trade = None

        # 🌙 ENTRY CONDITIONS
        if not self.position:
            # Long entry
            if (self.data.Close[-2] < self.bb_lower[-2]) and (self.data.Close[-1] > self.bb_lower[-1]) and (self.macd_line[-1] < self.macd_signal_line[-1]):  # Bullish crossover
                entry_price = self.data.Close[-1]
                stop_loss = min(self.swing_low[-1], self.bb_lower[-1])
                risk = entry_price - stop_loss
                
                if risk > 0:
                    size = int(round((self.equity * self.risk_percent) / risk))
                    if size > 0:
                        tp1 = entry_price + (self.bb_upper[-1] - self.bb_middle[-1])
                        tp2 = entry_price + 2 * (self
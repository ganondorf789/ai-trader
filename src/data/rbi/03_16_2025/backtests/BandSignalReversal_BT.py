```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR BANDSIGNALREVERSAL STRATEGY

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

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
        self.macd_signal = self.I(talib.MACD, self.data.Close, self.macd_fast, self.macd_slow, self.macd_signal, which=1)
        
        # Swing high/low using MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        self.active_trade = None

    def next(self):
        # 🌙 MOON DEV DEBUG: Print current indicators
        #print(f"🌙 Close: {self.data.Close[-1]:.2f} | BB_U: {self.bb_upper[-1]:.2f} | BB_L: {self.bb_lower[-1]:.2f}")
        
        # 🛑 EXIT CONDITIONS FIRST
        if self.position:
            current_close = self.data.Close[-1]
            
            if self.position.is_long:
                # Long exit conditions
                if (current_close >= self.active_trade['tp1']) or (current_close >= self.active_trade['tp2']):
                    print(f"🚀✨ MOON DEV TP HIT! Closing long at {current_close:.2f}")
                    self.position.close()
                elif crossunder(self.data.Close, self.bb_lower)[-1]:
                    print(f"🌙❌ STOP! Long closed at {current_close:.2f}")
                    self.position.close()
            
            elif self.position.is_short:
                # Short exit conditions
                if (current_close <= self.active_trade['tp1']) or (current_close <= self.active_trade['tp2']):
                    print(f"🚀✨ MOON DEV TP HIT! Closing short at {current_close:.2f}")
                    self.position.close()
                elif crossover(self.data.Close, self.bb_upper)[-1]:
                    print(f"🌙❌ STOP! Short closed at {current_close:.2f}")
                    self.position.close()
            self.active_trade = None

        # 🌙 ENTRY CONDITIONS
        if not self.position:
            # Long entry
            if crossover(self.data.Close, self.bb_lower) and self.macd_line[-1] < self.macd_signal[-1]:
                entry_price = self.data.Close[-1]
                stop_loss = min(self.swing_low[-1], self.bb_lower[-1])
                risk = entry_price - stop_loss
                
                if risk > 0:
                    size = int(round((self.equity * self.risk_percent) / risk))
                    self.buy(size=size)
                    current_std = (self.bb_upper[-1] - self.bb_middle[-1])/2
                    self.active_trade = {
                        'type': 'long',
                        'tp1': entry_price + current_std,
                        'tp2': entry_price + 2*current_std
                    }
                    print(f"🌙🚀 LONG ENTRY @ {entry_price:.2f} | Size: {size} | SL: {stop_loss:.2f}")
            
            # Short entry
            elif crossunder(self.data.Close, self.bb_upper) and self
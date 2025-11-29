I'll complete and debug the VolatilitySurgeReversal strategy code while maintaining all the original logic and adding Moon Dev themed debug prints. Here's the fixed version:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import time

class VolatilitySurgeReversal(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade
    atr_multiplier = 2
    rsi_period = 14
    std_dev_period = 20
    volume_ma_period = 20
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        self.std_dev = self.I(talib.STDDEV, self.data.Close, timeperiod=self.std_dev_period)
        self.mean_close = self.I(talib.SMA, self.data.Close, timeperiod=self.std_dev_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Track open interest changes
        self.open_interest = self.I(lambda x: x, self.data.OpenInterest)
        
        # Track entry conditions
        self.entry_price = None
        self.entry_low = None
        self.highest_high = None
        self.trailing_stop = None

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        current_oi = self.open_interest[-1]
        prev_oi = self.open_interest[-2] if len(self.open_interest) > 1 else current_oi
        
        # Moon Dev debug prints 🌙
        print(f"\n🌙 Moon Dev Debug - Bar: {self.data.index[-1]}")
        print(f"   Close: {current_close:.2f}, Vol: {current_volume:.2f}, OI: {current_oi} → {current_oi < prev_oi}")
        print(f"   3σ Level: {self.mean_close[-1] - 3*self.std_dev[-1]:.2f}")
        print(f"   RSI: {self.rsi[-1]:.2f}, ATR: {self.atr[-1]:.2f}")

        # Time filter (9:30am-3:30pm)
        current_time = self.data.index[-1].time()
        if current_time < time(9, 30) or current_time > time(15, 30):
            print("🌙 Moon Dev Note: Outside trading hours - skipping")
            return

        # Check if we're in position
        if self.position:
            # Update trailing stop logic
            self.highest_high = max(self.highest_high, self.data.High[-1])
            self.trailing_stop = self.highest_high - self.atr[-1] * self.atr_multiplier
            
            print(f"🚀 Position Active! Entry: {self.entry_price:.2f}")
            print(f"   Trailing Stop: {self.trailing_stop:.2f}, Current High: {self.highest_high:.2f}")

            # Exit conditions
            if self.rsi[-2] < 50 and self.rsi[-1] > 50:  # RSI crossover above 50
                print(f"✨ RSI Exit Signal! Closing position at {current_close:.2f}")
                self.position.close()
            elif self.data.Low[-1] < self.entry_low:
                print(f"🚨 Emergency Exit! Price made new low at {self.data.Low[-1]:.2f}")
                self.position.close()
            elif self.data.Close[-1] < self.trailing_stop:
                print(f"🔻 Trailing Stop Hit! Closing at {self.data.Close[-1]:.2f}")
                self.position.close()
            return

        # Entry conditions
        price_deviation = current_close < (self.mean_close[-1] - 3 * self.std_dev[-1])
        volume_surge = current_volume > (1.5 * self.volume_ma[-1])
        oi_decline = current_oi < prev_oi
        
        if price_deviation and volume_surge and oi_decline:
            print("🌙✨ ENTRY CONDITIONS MET! ✨🌙")
            print(f"   Price Dev: {price
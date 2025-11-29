```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

def upper_band(close, high, low, timeperiod_sma, timeperiod_atr, multiplier):
    sma = talib.SMA(close, timeperiod_sma)
    atr = talib.ATR(high, low, close, timeperiod_atr)
    return sma + multiplier * atr

def lower_band(close, high, low, timeperiod_sma, timeperiod_atr, multiplier):
    sma = talib.SMA(close, timeperiod_sma)
    atr = talib.ATR(high, low, close, timeperiod_atr)
    return sma - multiplier * atr

class VolumetricBreakout(Strategy):
    def init(self):
        # Calculate indicators using TA-Lib wrapped in self.I()
        self.upper = self.I(upper_band, self.data.Close, self.data.High, self.data.Low, 20, 14, 2.5)
        self.lower = self.I(lower_band, self.data.Close, self.data.High, self.data.Low, 20, 14, 2.5)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
        # Initialize tracking variables
        self.entry_atr = None
        self.highest_high = None
        self.lowest_low = None
        self.entry_bar = 0

    def next(self):
        current_bar = len(self.data) - 1
        
        # Moon Dev Risk Management Calculations 🌙
        risk_percent = 0.01
        equity = self.equity
        
        # Entry Logic
        if not self.position:
            # Long Entry 🌕
            if self.data.Close[-1] > self.upper[-1] and self.data.Volume[-1] > self.volume_ma[-1]:
                atr_value = self.atr[-1]
                if atr_value == 0:
                    return
                risk_amount = equity * risk_percent
                position_size = int(round(risk_amount / (1.5 * atr_value)))
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_atr = atr_value
                    self.highest_high = self.data.High[-1]
                    self.entry_bar = current_bar
                    print(f"🚀 LONG ENTRY! Price: {self.data.Close[-1]:.2f}, Size: {position_size} | Moon Dev launching rockets! 🌕")

            # Short Entry 🌑
            elif self.data.Close[-1] < self.lower[-1] and self.data.Volume[-1] > self.volume_ma[-1]:
                atr_value = self.atr[-1]
                if atr_value == 0:
                    return
                risk_amount = equity * risk_percent
                position_size = int(round(risk_amount / (1.5 * atr_value)))
                if position_size > 0:
                    self.sell(size=position_size)
                    self.entry_atr = atr_value
                    self.lowest_low = self.data.Low[-1]
                    self.entry_bar = current_bar
                    print(f"🌑 SHORT ENTRY! Price: {self.data.Close[-1]:.2f}, Size: {position_size} | Moon Dev shadow trading! 🌘")

        # Exit Logic
        else:
            # Long Position Management 🌕
            if self.position.is_long:
                # Update trailing stop
                self.highest_high = max(self.highest_high, self.data.High[-1])
                stop_price = self.highest_high - 1.5 * self.entry_atr
                
                # RSI Exit
                if crossunder(self.rsi.array[-2:], 70):
                    self.position.close()
                    print(f"✨ LONG EXIT! RSI<70 | Moon Dev taking lunar profits! 🌓")
                
                # Time Exit
                elif current_bar - self.entry_bar >= 480:  # 5 days in 15m bars
                    self.position.close()
                    print(f"⏳ TIME EXIT! 5 days passed | Moon Dev cycle complete! 🌗")
                
                # Stop Exit
                elif self.data.Low[-1] <= stop_price:
                    self.position
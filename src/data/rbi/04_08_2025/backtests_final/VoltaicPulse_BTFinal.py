I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
from backtesting import Backtest, Strategy
import talib

class VoltaicPulse(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    trailing_stop_multiplier = 2  # 2x ATR for trailing stop
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[
            col for col in self.data.df.columns if 'unnamed' in col
        ], errors='ignore')  # Added errors='ignore' for safety
        
        # Calculate indicators using TA-Lib with self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.ema_atr = self.I(talib.EMA, self.atr, timeperiod=20)
        
        # Check if VIX column exists before using it
        if 'vix' in self.data.df.columns:
            self.vix = self.I(lambda x: x, self.data.df['vix'])
            self.sma_vix = self.I(talib.SMA, self.vix, timeperiod=20)
        else:
            print("🌙 WARNING: VIX data not found in dataframe! ✨")
            self.vix = self.sma_vix = None
        
        # Initialize tracking variables
        self.entry_signal = False
        self.highest_high = None
        self.entry_price = None

    def next(self):
        # Skip if we don't have enough data or VIX is missing
        if len(self.atr) < 2 or len(self.ema_atr) < 2 or self.vix is None:
            return
            
        if not self.position and not self.entry_signal:
            # Check for ATR crossover condition
            atr_cross = (self.atr[-2] < self.ema_atr[-2]) and (self.atr[-1] > self.ema_atr[-1])
            vix_condition = (self.vix[-1] < self.sma_vix[-1])
            
            if atr_cross and vix_condition:
                print(f"🌙 VOLTAICPULSE TRIGGERED! ATR({self.atr[-1]:.2f}) > EMA_ATR({self.ema_atr[-1]:.2f}) ✨")
                print(f"🌙 VIX({self.vix[-1]:.2f}) < SMA_VIX({self.sma_vix[-1]:.2f}) - Sentiment GO! 🚀")
                self.entry_signal = True

        if self.entry_signal and not self.position:
            # Calculate position size with risk management
            atr_value = self.atr[-1]
            stop_distance = self.trailing_stop_multiplier * atr_value
            risk_amount = self.risk_per_trade * self.equity
            position_size = int(round(risk_amount / stop_distance))
            
            if position_size > 0:
                self.buy(size=position_size)
                self.highest_high = self.data.Close[-1]
                self.entry_price = self.data.Close[-1]
                print(f"🌙 MOON DEV ENTRY 🌙 Size: {position_size} @ {self.data.Close[-1]:.2f}")
                print(f"🌙 Initial Stop: {self.entry_price - stop_distance:.2f} (Risk: {stop_distance:.2f}) ⚡")
                self.entry_signal = False

        if self.position:
            # Update trailing stop
            current_high = self.data.High[-1]
            if current_high > self.highest_high:
                self.highest_high = current_high
                print(f"🌙 New Swing High: {self.highest_high:.2f} 🎯")

            stop_level = self.highest_high - (self.trailing_stop_multiplier * self.atr[-1])
            
            if self.data.Low[-1] <= stop_level:
                self.position.close()
                print(f"🌙 TRAILING STOP HIT 🌙 Exit @ {stop_level:.2f} 💥")
                print(f"🌙 Trade P/L: {self.position.pl:.2f} 🌕\n")

# Data preparation
try:
    data = pd.read_csv('your_data_file
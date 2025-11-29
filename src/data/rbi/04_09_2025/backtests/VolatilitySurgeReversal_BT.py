```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# Data preparation function
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Ensure proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'openinterest': 'OpenInterest'
    })
    # Convert datetime column
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

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

        # Check if we're in position
        if self.position:
            # Update trailing stop logic
            self.highest_high = max(self.highest_high, self.data.High[-1])
            self.trailing_stop = self.highest_high - self.atr[-1] * self.atr_multiplier
            
            print(f"🚀 Position Active! Entry: {self.entry_price:.2f}")
            print(f"   Trailing Stop: {self.trailing_stop:.2f}, Current High: {self.highest_high:.2f}")

            # Exit conditions
            if crossover(self.rsi, 50):
                print(f"✨ RSI Exit Signal! Closing position at {current_close:.2f}")
                self.position.close()
            elif self.data.Low[-1] < self.entry_low:
                print(f"🚨 Emergency Exit! Price made new low at {self.data.Low[-1]:.2f}")
                self.position.close()
            elif self.data.Close[-1] < self.trailing_stop:
                print(f"🔻 Trailing Stop Hit! Closing at {self.data.Close[-1]:.2f}")
                self.position.close()
            return

        # Time filter check (9:30am-3:30pm EST, exclude first/last 30 mins)
        current_time = self.data.index[-1].time()
        session_start = pd.Timestamp('09:30').time()
        session_end = pd.Timestamp('15:30').time()
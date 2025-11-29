Here's the complete fixed code with Moon Dev themed debug prints and necessary technical fixes:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation
def load_data():
    data = pd.read_csv(
        '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
        parse_dates=['datetime'],
        index_col='datetime'
    )
    # Clean data columns
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Rename to proper case
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

class FibStochTrend(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    fib_tolerance = 0.005  # 0.5% tolerance for Fib levels
    swing_period = 20
    stoch_fastk = 14
    
    def init(self):
        # Core Indicators 🌙
        print("🌙 Initializing Moon Dev Indicators...")
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        
        # Stochastic Oscillator ✨
        def calculate_stoch():
            slowk, slowd = talib.STOCH(
                self.data.High, self.data.Low, self.data.Close,
                fastk_period=self.stoch_fastk,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            return slowk, slowd
        self.slowk, self.slowd = self.I(calculate_stoch, name=['SlowK', 'SlowD'])
        print("✨ Stochastic Oscillator initialized successfully!")
        
    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        if len(self.data.Close) < 200:  # Wait for indicators to warm up
            print("🌙 Waiting for indicators to warm up...")
            return
            
        # Trend Analysis 🌊
        trend_up = current_close > self.sma200[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        # Fibonacci Calculations 📐
        if trend_up and swing_high > swing_low:
            fib_range = swing_high - swing_low
            fib_levels = [
                swing_high - fib_range * 0.382,
                swing_high - fib_range * 0.5,
                swing_high - fib_range * 0.618
            ]
        elif not trend_up and swing_high > swing_low:
            fib_range = swing_high - swing_low
            fib_levels = [
                swing_low + fib_range * 0.382,
                swing_low + fib_range * 0.5,
                swing_low + fib_range * 0.618
            ]
        else:
            return
            
        # Check Fib retracement 📍
        near_fib = any(
            (fib * (1 - self.fib_tolerance) <= current_close <= fib * (1 + self.fib_tolerance)
            for fib in fib_levels
        )
        
        # Stochastic Conditions 📈
        slowk = self.slowk[-1]
        prev_slowk = self.slowk[-2] if len(self.slowk) > 2 else 0
        
        if trend_up:
            stoch_signal = prev_slowk <= 20 and slowk > 20
        else:
            stoch_signal = prev_slowk >= 80 and slowk < 80
            
        # Entry Logic 🚀
        if not self.position and near_fib and stoch_signal:
            # Risk Management 🔒
            if trend_up:
                stop_loss = swing_low
                risk_per_share = current_close - stop_loss
                position_size = self.risk_percent
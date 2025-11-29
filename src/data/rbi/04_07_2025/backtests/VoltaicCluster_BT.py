```python
# -*- coding: utf-8 -*-
import talib
import pandas as pd
import numpy as np
import pandas_ta as pta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Standardize column names for backtesting.py
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicCluster(Strategy):
    # Strategy parameters
    chaikin_period = 10
    volatility_threshold = 0.5
    fib_level_exit = 1.618
    risk_percent = 0.01
    atr_period = 14
    delta_window = 20
    
    def init(self):
        # 🌙 Initialize indicators with self.I()
        self.chaikin = self.I(self.calculate_chaikin)
        self.delta = self.I(self.calculate_delta)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Track trade parameters
        self.stop_loss = None
        self.take_profit = None
        self.entry_price = None
        
    def calculate_chaikin(self):
        """🌙 Chaikin Volatility Indicator"""
        hl_range = self.data.High - self.data.Low
        ema1 = talib.EMA(hl_range, timeperiod=self.chaikin_period)
        ema2 = talib.EMA(ema1, timeperiod=self.chaikin_period*2)
        return ((ema1 - ema2) / ema2) * 100  # Percentage volatility change
    
    def calculate_delta(self):
        """🌙 Mock Order Book Delta (Simulated Imbalance)"""
        # Using RSI as temporary placeholder for delta calculation
        return pta.rsi(self.data.Close, length=self.delta_window) - 50  # Centered at zero
    
    def next(self):
        # 🌙✨ Moon Dev Safety Check: Ensure indicators are valid
        if len(self.data) < 50 or np.isnan(self.chaikin[-1]):
            return
        
        current_price = self.data.Close[-1]
        fib_ext = self.entry_price * self.fib_level_exit if self.entry_price else 0
        
        # 🚀 Risk Management Calculations
        equity = self.equity
        risk_amount = equity * self.risk_percent
        atr_value = self.atr[-1] if self.atr[-1] > 0 else current_price * 0.01
        point_value = 1  # Adjust based on your market
        
        # 🌙 Position Sizing Calculation
        position_size = int(round(risk_amount / (atr_value * point_value)))
        position_size = max(1, position_size)  # Minimum 1 unit
        
        # Exit Conditions
        if self.position:
            # 🌙 Trailing Stop Logic
            if self.chaikin[-1] < self.chaikin[-2] * 0.9:
                print(f"🌙✨ Chaikin Volatility Contracting! Preparing to exit...")
                self.position.close()
                
            # Fibonacci Exit Logic
            if (self.position.is_long and current_price >= fib_ext) or \
               (self.position.is_short and current_price <= fib_ext):
                print(f"🚀💎 Reached Fib Extension at {fib_ext:.2f}! Taking profits!")
                self.position.close()
        
        # Entry Conditions
        if not self.position:
            # Long Entry: Chaikin Expansion + Bullish Delta
            if self.chaikin[-1] > self.volatility_threshold and self.delta[-1] > 5:
                print(f"🌙🚀 BULLISH CLUSTER DETECTED! Buying {position_size} units")
                self.buy(size=position_size)
                self.entry_price = current_price
                
            # Short Entry: Chaikin Expansion + Bearish Delta
            elif self.chaikin
```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class DynamicDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 14
    rsi_period = 14
    swing_period = 5
    sma_period = 20
    
    def init(self):
        # Calculate indicators using TA-Lib
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        
        # Swing highs/lows using MAX/MIN
        self.swing_highs = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_lows = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
    def next(self):
        current_idx = len(self.data)-1
        if current_idx < max(self.swing_period, self.atr_period, self.rsi_period) + 1:
            return
        
        # Get recent values
        price_low = self.data.Low[-1]
        prev_low = self.data.Low[-2]
        rsi_current = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        
        price_high = self.data.High[-1]
        prev_high = self.data.High[-2]
        
        # Moon Dev Debug Prints 🌙
        print(f"🌙 Current Close: {self.data.Close[-1]:.2f} | RSI: {rsi_current:.2f} | SMA: {self.sma[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        
        # Bullish Divergence Detection (Price Lower Low + RSI Higher Low)
        bullish_div = (price_low < prev_low) and (rsi_current > rsi_prev)
        
        # Bearish Divergence Detection (Price Higher High + RSI Lower High)
        bearish_div = (price_high > prev_high) and (rsi_current < rsi_prev)
        
        # Entry Conditions
        if not self.position:
            # Long Entry: Bullish Div + Close > SMA + RSI Cross Above 50
            if (bullish_div and 
                self.data.Close[-1] > self.sma[-1] and 
                rsi_current > 50 and rsi_prev <= 50):
                
                # Risk Management Calculations
                atr_value = self.atr[-1]
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - 1.5 * atr_value
                take_profit = entry_price + 3 * atr_value  # 2:1 RR
                
                # Position Sizing
                risk_amount = self.equity * self.risk_percent
                position_size = risk_amount / (entry_price - stop_loss)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    print(f"🚀🌙 BULLISH DIVERGENCE CONFIRMED! Long Entry at {entry_price:.2f}")
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            
            # Short Entry: Bearish Div + Close < SMA + RSI Cross Below 50
            elif (bearish_div and 
                  self.data.Close[-1] < self.sma[-1] and 
                  rsi_current < 50 and rsi_prev >= 50):
                  
                atr_value = self.atr[-1]
                entry_price = self.data.Close[-1
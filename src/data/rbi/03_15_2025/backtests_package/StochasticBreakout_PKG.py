Here's the fixed code with all `backtesting.lib` imports and functions removed, and replaced with proper Moon Dev themed implementations:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest

class StochasticBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    take_profit_pct = 0.02  # 2% take profit
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])

        # Calculate indicators using TA-Lib with self.I()
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(self.data.High, self.data.Low, self.data.Close,
                                      fastk_period=14,
                                      slowk_period=3,
                                      slowk_matype=0,
                                      slowd_period=3,
                                      slowd_matype=0)
        self.stoch_k = self.I(lambda: stoch_k, name='Stoch %K')
        self.stoch_d = self.I(lambda: stoch_d, name='Stoch %D')

        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')

        # Support/Resistance using 20-period swing levels
        self.resistance = self.I(talib.MAX, self.data.High, 20, name='Resistance')
        self.support = self.I(talib.MIN, self.data.Low, 20, name='Support')

    def next(self):
        if self.position:
            return  # Exit if position already exists

        # Get current values
        close = self.data.Close[-1]
        stoch_k = self.stoch_k[-1]
        stoch_d = self.stoch_d[-1]
        rsi = self.rsi[-1]
        resistance = self.resistance[-1]
        support = self.support[-1]

        # Long entry conditions 🌙
        long_trigger = (
            (self.stoch_k[-2] < self.stoch_d[-2] and self.stoch_k[-1] > self.stoch_d[-1]) and  # Bullish crossover
            stoch_k < 20 and
            rsi > 50 and
            close > resistance
        )

        # Short entry conditions 🌙
        short_trigger = (
            (self.stoch_k[-2] > self.stoch_d[-2] and self.stoch_k[-1] < self.stoch_d[-1]) and  # Bearish crossover
            stoch_k > 80 and
            rsi < 50 and
            close < support
        )

        # Execute trades with proper position sizing
        if long_trigger:
            self.enter_long(close, resistance)
            
        elif short_trigger:
            self.enter_short(close, support)

    def enter_long(self, entry_price, stop_level):
        risk_per_share = entry_price - stop_level
        if risk_per_share <= 0: return
        
        position_size = self.calculate_position_size(risk_per_share)
        take_profit = entry_price * (1 + self.take_profit_pct)
        
        if position_size > 0:
            self.buy(size=position_size, sl=stop_level, tp=take_profit)
            print(f"🌙✨ Moon Dev LONG Signal! 🚀 Entry: {entry_price:.2f}")
            print(f"   Size: {position_size}, SL: {stop_level:.2f}, TP: {take_profit:.2f}")

    def enter_short(self, entry_price, stop_level):
        risk_per_share = stop_level - entry_price
        if risk_per_share <= 0: return
        
        position_size = self.calculate_position_size(risk_per_share)
        take_profit = entry_price * (1 - self.take_profit_pct)
        
        if position_size > 0:
            self.sell(size=position_size, sl=stop_level, tp=take_profit)
            print(f"🌙✨ Moon Dev SHORT Signal! 🌧️ Entry: {entry_price:.2f}")
            print(f"   Size: {position_size}, SL: {stop_level:.2f}, TP: {take_profit:.2f}")

    def calculate_position_size(self, risk_per_share):
        risk_amount = self.equity * self.risk_percent
        position_size = risk_amount / risk_per_share
        return int(round(position_size))

# Data preparation
data
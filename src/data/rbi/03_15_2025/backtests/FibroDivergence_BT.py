```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class FibroDivergence(Strategy):
    rsi_period = 14
    swing_period = 20
    atr_period = 14
    risk_pct = 0.01
    fib_levels = [0.382, 0.5, 0.618]
    
    def init(self):
        # Core indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Debug counters
        self.bullish_signals = 0
        self.bearish_signals = 0

    def next(self):
        price = self.data.Close[-1]
        
        # Avoid trading during first 50 periods
        if len(self.data) < 50 or not all([self.rsi[-3:], self.swing_high[-3:], self.swing_low[-3:]]):
            return
            
        # Calculate dynamic Fibonacci levels
        current_swing_high = self.swing_high[-1]
        current_swing_low = self.swing_low[-1]
        fib_levels = [current_swing_low + (current_swing_high - current_swing_low) * level for level in self.fib_levels]
        
        # Detect divergence
        bullish_div = False
        bearish_div = False
        
        # Bullish divergence check (price lower low, RSI higher low)
        if (self.data.Low[-1] < self.data.Low[-3] and
            self.rsi[-1] > self.rsi[-3] and
            self.rsi[-3] < 45):
            self.bullish_signals += 1
            bullish_div = True
            print(f"🌙✨ Bullish divergence detected! RSI: {self.rsi[-3]:.1f}→{self.rsi[-1]:.1f} Price: {self.data.Low[-3]:.1f}→{self.data.Low[-1]:.1f}")
        
        # Bearish divergence check (price higher high, RSI lower high)
        if (self.data.High[-1] > self.data.High[-3] and
            self.rsi[-1] < self.rsi[-3] and
            self.rsi[-3] > 55):
            self.bearish_signals += 1
            bearish_div = True
            print(f"🌙✨ Bearish divergence detected! RSI: {self.rsi[-3]:.1f}→{self.rsi[-1]:.1f} Price: {self.data.High[-3]:.1f}→{self.data.High[-1]:.1f}")
        
        # Check Fibonacci retracement levels
        fib_active = any(abs(price - level) < 0.01 * price for level in fib_levels)
        
        # Entry logic with Moon Dev flair
        if not self.position:
            # Long entry conditions
            if bullish_div and fib_active and price > min(fib_levels):
                risk_amount = self.risk_pct * self.equity
                sl = price - 1.5 * self.atr[-1]
                position_size = int(round(risk_amount / (price - sl)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tag="Moon Bullish Entry")
                    print(f"🚀🌙 LONG launched! Size: {position_size} | Entry: {price:.2f} | SL: {sl:.2f} |
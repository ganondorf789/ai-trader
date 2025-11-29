Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev's FibroPullback Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# 🚀 Data Preparation 🚀
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class FibroPullback(Strategy):
    # 🌟 Strategy Parameters 🌟
    risk_per_trade = 0.01  # 1% risk per trade
    swing_period = 20
    ema50_period = 50
    ema200_period = 200
    fib_retracement = 0.618
    fib_sl_level = 0.786  # 78.6% for stop loss
    fib_tp_level = 1.618  # 161.8% extension
    
    def init(self):
        # 🌌 Indicator Calculations 🌌
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period)
        
        # 🕯️ Candlestick Patterns 🕯️
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.shooting_star = self.I(talib.CDLSHOOTINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        # 🌓 Moon Dev's Core Logic 🌓
        if len(self.data) < 200:  # Wait for enough data
            return
            
        # Current values
        close = self.data.Close[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        
        # 🌗 Trend Detection 🌗
        uptrend = ema50 > ema200
        
        # 🌉 Fibonacci Calculations 🌉
        if uptrend:
            fib_level = swing_high - (swing_high - swing_low) * self.fib_retracement
            sl_level = swing_high - (swing_high - swing_low) * self.fib_sl_level
            tp_level = swing_high + (swing_high - swing_low) * (self.fib_tp_level - 1)
        else:
            fib_level = swing_low + (swing_high - swing_low) * self.fib_retracement
            sl_level = swing_low + (swing_high - swing_low) * self.fib_sl_level
            tp_level = swing_low - (swing_high - swing_low) * (self.fib_tp_level - 1)
        
        # 🎯 Entry Conditions 🎯
        price_near_fib = abs(close - fib_level) < 0.005 * fib_level  # 0.5% tolerance
        
        # Long Entry
        if uptrend and price_near_fib and not self.position:
            if self.hammer[-1] > 0 or self.engulfing[-1] > 0:
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = close - sl_level
                if risk_per_share > 0:
                    position_size = int(round(risk_amount /
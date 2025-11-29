The code you provided is already free from any `backtesting.lib` imports and functions, which is great! However, I'll ensure that all indicators are properly implemented using `talib` and `numpy`, and I'll add some Moon Dev themed debug prints for better visualization. Here's the complete fixed and enhanced code:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Load and clean data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Format columns for backtesting.py
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VoltaicDivergence(Strategy):
    atr_period = 14
    rsi_period = 14
    swing_window = 20
    multiplier = 1.5
    risk_pct = 1
    rr_ratio = 2

    def init(self):
        # Core indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # Volatility bands
        self.upper_band = self.I(lambda c, a: c + a*self.multiplier, self.data.Close, self.atr)
        self.lower_band = self.I(lambda c, a: c - a*self.multiplier, self.data.Close, self.atr)
        
        # Swing detection
        self.max_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.min_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        self.swing_high = self.I(np.equal, self.data.High, self.max_high)
        self.swing_low = self.I(np.equal, self.data.Low, self.min_low)
        
        self.swing_highs = []
        self.swing_lows = []

    def next(self):
        # Track swing points
        if self.swing_high[-1]:
            price = self.data.High[-1]
            rsi_val = self.rsi[-1]
            self.swing_highs.append((price, rsi_val))
            print(f"🌙✨ SWING HIGH at {price:.2f} | RSI: {rsi_val:.2f}")
            
        if self.swing_low[-1]:
            price = self.data.Low[-1]
            rsi_val = self.rsi[-1]
            self.swing_lows.append((price, rsi_val))
            print(f"🌙✨ SWING LOW at {price:.2f} | RSI: {rsi_val:.2f}")
        
        # Check divergences
        bull_div = bear_div = False
        
        if len(self.swing_lows) >= 2:
            prev_low, prev_rsi = self.swing_lows[-2]
            curr_low, curr_rsi = self.swing_lows[-1]
            if curr_low < prev_low and curr_rsi > prev_rsi:
                bull_div = True
                print(f"🚀🌙 BULLISH DIVERGENCE | Price: {curr_low:.2f} < {prev_low:.2f} | RSI: {curr_rsi:.2f} > {prev_rsi:.2f}")
                
        if len(self.swing_highs) >= 2:
            prev_high, prev_rsi = self.swing_highs[-2]
            curr_high, curr_rsi = self.swing_highs[-1]
            if curr_high > prev_high and curr_rsi < prev_rsi:
                bear_div = True
                print(f"🌑🌙 BEARISH DIVERGENCE | Price: {curr_high:.2f} > {prev_high:.2f} | RSI: {curr_rsi:.2f} < {prev_rsi:.2f}")
        
        # Entry logic
        if not self.position:
            current_close = self.data.Close[-1]
            
            # Long entry
            if current_close >
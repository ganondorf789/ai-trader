Here's the complete implementation of the SqueezeCascade strategy for backtesting.py:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Validate and rename columns
required_cols = {'open', 'high', 'low', 'close', 'volume', 'funding_rate'}
if not required_cols.issubset(data.columns):
    missing = required_cols - set(data.columns)
    raise ValueError(f"Missing required columns: {missing}")

data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data.index = pd.to_datetime(data['datetime'])
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'funding_rate']]

class SqueezeCascade(Strategy):
    funding_threshold = 0.0001  # 0.01%
    bandwidth_ratio = 0.5
    risk_pct = 0.01  # 1% per trade
    
    def init(self):
        # Bollinger Bands
        close = self.data.Close
        self.bb_upper, self.bb_middle, self.bb_lower = (
            self.I(lambda: talib.BBANDS(close, 20, 2, 2)[0], 
            self.I(lambda: talib.BBANDS(close, 20, 2, 2)[1], 
            self.I(lambda: talib.BBANDS(close, 20, 2, 2)[2])
        
        # Bandwidth calculations
        bb_width = self.bb_upper - self.bb_lower
        self.bandwidth = self.I(lambda: bb_width / self.bb_middle)
        self.bandwidth_avg = self.I(lambda: talib.SMA(self.bandwidth, 20))
        
        # Volatility and volume indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
    def next(self):
        if self.position:
            current_low = self.data.Low[-1]
            current_high = self.data.High[-1]
            
            # Check profit targets
            if current_low <= self.tp1:
                self.position.close(0.5)
                print(f"🌙✨ TP1 Hit! Closed 50% at {current_low:.2f}")
                
            if current_low <= self.swing_low[-1]:
                self.position.close()
                print(f"🚀📉 Swing Low Reached! Full exit at {current_low:.2f}")
                
            # Trailing stop logic
            if current_low <= self.trail_trigger and not self.trailing:
                print(f"🛡️🌙 Trailing Stop Activated!")
                self.trailing = True
                self.orders[0].sl = self.entry_price  # Update existing SL
                
            # Stop loss check
            if current_high >= self.sl_price:
                self.position.close()
                print(f"🌙💥 SL Triggered at {current_high:.2f}")
                
            return
            
        # Entry conditions
        if (self.data.funding_rate[-1] >= self.funding_threshold and
            self.bandwidth[-1] < self.bandwidth_avg[-1] * self.bandwidth_ratio and
            self.data.Close[-1] < self.bb_lower[-1] and
            self.data.Volume[-1] > self.volume_sma[-1]):
            
            # Risk calculations
            self.entry_price = self.data.Close[-1]
            self.sl_price = self.bb_upper[-1]
            risk_per_unit = self.sl_price - self.entry_price
            if risk_per_unit <= 0: return
            
            position_size = int(round((self.equity * self.risk_pct) / risk_per_unit))
            if position_size < 1: return
            
            # Profit targets
            self.tp1 =
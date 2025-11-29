I'll analyze and fix the code to ensure no backtesting.lib imports are used and all indicators are properly implemented. Here's the corrected version with Moon Dev themed debug prints:

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
    raise ValueError(f"🌙❌ Missing required columns: {missing}")

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
        # Bollinger Bands using talib
        close = self.data.Close
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_middle = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        
        # Bandwidth calculations
        bb_width = self.bb_upper - self.bb_lower
        self.bandwidth = self.I(lambda: bb_width / self.bb_middle)
        self.bandwidth_avg = self.I(talib.SMA, self.bandwidth, timeperiod=20)
        
        # Volatility and volume indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        print("🌙✨ Strategy initialized with Moon Dev indicators!")
        
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
            self.sl_price = self.
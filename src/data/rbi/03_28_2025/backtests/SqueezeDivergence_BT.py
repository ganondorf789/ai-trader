To address this problem, we need to create a backtesting script for the SqueezeDivergence strategy using the backtesting.py framework. The strategy involves identifying Bollinger Band squeezes combined with volume divergence to enter counter-trend trades and exit based on volatility expansion or time-based conditions. Here's the step-by-step solution:

### Approach
1. **Data Handling**: Clean and preprocess the data to ensure proper column names and structure.
2. **Indicators Calculation**: Use TA-Lib to compute Bollinger Bands, ATR, and other necessary indicators.
3. **Entry Conditions**: Identify Bollinger Band squeeze conditions and volume divergence to trigger trades.
4. **Exit Conditions**: Exit trades based on volatility expansion or after a fixed number of bars.
5. **Risk Management**: Calculate position sizes based on a 1% risk per trade and set stop losses using ATR.
6. **Moon Dev Themed Logging**: Include themed print statements for debugging and tracking trades.

### Solution Code
```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class SqueezeDivergence(Strategy):
    initial_capital = 1_000_000  # Set initial capital to 1,000,000
    
    def init(self):
        # Calculate Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name=['BB_Upper', 'BB_Middle', 'BB_Lower'])
        
        # Calculate Bollinger Band Width
        self.bb_width = self.I(lambda u, l, m: ((u - l) / m) * 100, self.bb_upper, self.bb_lower, self.bb_middle, name='BB_Width')
        
        # Calculate 1-year (35000 periods) 5th percentile for squeeze detection
        self.bb_squeeze_threshold = self.I(lambda x: x.rolling(35000).quantile(0.05), self.bb_width, name='BB_Squeeze_Threshold')
        
        # Calculate Swing Highs for price and volume
        self.price_swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Price_Swing_High')
        self.volume_swing_high = self.I(talib.MAX, self.data.Volume, timeperiod=20, name='Volume_Swing_High')
        
        # Calculate ATR for stop loss
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        # Calculate 50-period SMA of BB Width for exit condition
        self.bb_width_ma50 = self.I(talib.SMA, self.bb_width, timeperiod=50, name='BB_Width_MA50')
        
        print("🌙✨ Moon Dev Indicators Initialized! Ready to launch! 🚀")

    def next(self):
        # Check if we have enough data to compute all indicators
        if len(self.data) < 35000:
            return
        
        # Current values
        current_bb_width = self.bb_width[-1]
        squeeze_threshold = self.bb_squeeze_threshold[-1]
        squeeze_condition = current_bb_width < squeeze_threshold if not np.isnan(squeeze_threshold) else False
        
        # Check if we're in a squeeze
        if squeeze_condition:
            # Calculate divergence conditions
            current_price_high = self.data.High[-1]
            previous_price_swing_high = self.price_swing_high[-2]
            current_volume = self.data.Volume[-1]
            previous_volume_swing_high = self
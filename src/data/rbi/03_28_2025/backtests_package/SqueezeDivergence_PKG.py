I'll fix the code by removing all `backtesting.lib` imports and replacing any usage of its functions with appropriate alternatives. Here's the corrected version:

```python
from backtesting import Backtest, Strategy
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
            previous_volume_swing_high = self.volume_swing_high[-2]
            
            # Check for bearish divergence (price higher high but volume lower high)
            if (current_price_high > previous_price_swing_high and 
                current_volume < previous_volume_swing_high):
                # Calculate position size with 1% risk
                risk_amount = self.equity * 0.01
                stop_loss_distance = self.atr[-1] * 2
                position_size = risk_amount / stop_loss_distance
                
                # Enter short position
                self.sell(size=position_size, sl=current_price_high + stop_loss_distance)
                print(f"🌙✨ Moon Dev Short Signal Activated! Selling at {current_price_high:.2f} with SL at {current_price_high + stop_loss_distance:.2f}")
            
            # Check for bullish divergence (
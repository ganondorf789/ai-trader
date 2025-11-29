Here is the fixed code with all the necessary changes to remove any `backtesting.lib` imports and ensure proper indicator usage:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolumetricDivergence(Strategy):
    sma_period = 50
    volume_sma_period = 20
    swing_period = 20
    atr_period = 14
    distance_threshold = 0.01  # 1% price distance
    risk_percent = 0.01  # 1% risk per trade
    sl_multiplier = 1.5
    tp_multiplier = 2

    def init(self):
        # Core indicators 🌙
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_sma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Swing detection indicators ✨
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Track swing states
        self.last_swing_high_price = None
        self.last_swing_high_volume = None
        self.last_swing_low_price = None
        self.last_swing_low_volume = None

    def next(self):
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_close = self.data.Close[-1]

        # Skip if indicators not ready 🌙
        if len(self.data) < self.swing_period or np.isnan(self.swing_high[-1]):
            return

        # Check for new swing high (bullish divergence detection)
        if current_high == self.swing_high[-1]:
            self._process_swing_high(current_high)

        # Check for new swing low (bearish divergence detection)
        if current_low == self.swing_low[-1]:
            self._process_swing_low(current_low)

    def _process_swing_high(self, current_high):
        current_volume = self.data.Volume[-1]
        
        if self.last_swing_high_price and self.last_swing_high_volume:
            # Detect bullish divergence conditions 🚀
            price_higher = current_high > self.last_swing_high_price
            volume_lower = current_volume < self.last_swing_high_volume
            sma_trend = self.sma[-1] > self.sma[-2]

            if price_higher and volume_lower and sma_trend:
                distance_pct = (current_high - self.last_swing_high_price) / self.last_swing_high_price
                
                if distance_pct >= self.distance_threshold and not self.position:
                    self._enter_long(current_high)

        # Update swing high tracking ✨
        self.last_swing_high_price = current_high
        self.last_swing_high_volume = current_volume

    def _process_swing_low(self, current_low):
        current_volume = self.data.Volume[-1]
        
        if self.last_swing_low_price and self.last_swing_low_volume:
            # Detect bearish divergence conditions 🌙
            price_lower = current_low < self.last_swing_low_price
            volume_lower = current_volume < self.last_swing_low_volume
            sma_trend = self.sma[-1] < self.sma[-2]

            if price_lower and volume_lower and sma_trend:
                distance_pct = (self.last_swing_low
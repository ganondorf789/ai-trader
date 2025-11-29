```python
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolDivergenceSurge(Strategy):
    risk_percent = 0.01
    swing_period = 20
    lookback_divergence = 5
    
    def init(self):
        # Bollinger Bands components
        self.bb_upper = self.I(self._bb_upper, self.data.Close)
        self.bb_middle = self.I(self._bb_middle, self.data.Close)
        self.bb_lower = self.I(self._bb_lower, self.data.Close)
        
        # Bollinger Bandwidth calculations
        self.bb_bandwidth = self.I(self._calc_bandwidth, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_low = self.I(talib.MIN, self.bb_bandwidth, 10)
        self.bb_high = self.I(talib.MAX, self.bb_bandwidth, 10)
        
        # Elder Force Index
        self.efi = self.I(self._calc_efi, self.data.Close, self.data.Volume)
        
        # Volume threshold
        self.vol_threshold = self.I(self._calc_volume_threshold, self.data.Volume)
        
        # Trend SMA
        self.sma50 = self.I(talib.SMA, self.data.Close, 50)
        
        print("🌙 Lunar indicators activated! Ready for volatility surge detection 🚀")

    def _bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, 20, 2, 2)
        return upper

    def _bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, 20, 2, 2)
        return middle

    def _bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, 20, 2, 2)
        return lower

    def _calc_bandwidth(self, upper, lower, middle):
        return (upper - lower) / middle

    def _calc_efi(self, close, volume):
        force = (close - np.roll(close, 1)) * volume
        force[0] = 0  # Handle first element
        return talib.EMA(force, 13)

    def _calc_volume_threshold(self, volume):
        return ta.quantile(pd.Series(volume), 50, 0.9).values

    def next(self):
        if len(self.data) < 50:
            return

        # Entry conditions
        price_high = talib.MAX(self.data.High, self.lookback_divergence)[-1]
        efi_high = talib.MAX(self.efi, self.lookback_divergence)[-1]
        prev_price_high = talib.MAX(self.data.High, self.lookback_divergence)[-self.lookback_divergence-1]
        prev_efi_high = talib.MAX(self.efi, self.lookback_divergence)[-self.lookback_divergence-1]

        conditions = (
            self.bb_bandwidth[-1] <= self.bb_low[-1],  # Bandwidth contraction
            self.bb_bandwidth[-1] > self.bb_bandwidth[-2],  # Expansion trigger
            self.data.Volume[-1] > self.vol_threshold[-1],  # Volume surge
            (price_high > prev_price_high) and (efi_high < prev_efi_high),  # Divergence
            self.data.Close[-1] < self.sma50[-1]  # Downtrend
        )

        if all(conditions):
            # Risk management calculations
            swing_high = talib.MAX(self
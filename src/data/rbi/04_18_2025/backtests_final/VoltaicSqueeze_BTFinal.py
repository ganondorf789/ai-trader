Here's the fixed and completed version of the VoltaicSqueeze strategy with Moon Dev themed debug prints and proper position sizing:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# Moon Dev Data Loading 🌙
print("🌙 Loading celestial market data...")
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
print("🌙 Data purification complete! Ready for lunar analysis ✨")

class VoltaicSqueeze(Strategy):
    bb_period = 20
    bb_dev = 2
    bandwidth_threshold = 20
    volume_multiplier = 1.5
    risk_pct = 0.01
    swing_period = 5

    def init(self):
        # Moon Dev Indicator Initialization 🌙
        print("🌙 Initializing celestial indicators...")
        
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return upper
        self.upper = self.I(bb_upper, self.data.Close, name='UpperBB')

        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return lower
        self.lower = self.I(bb_lower, self.data.Close, name='LowerBB')

        # Bandwidth Percentile
        def calc_percentile(series):
            return ta.percentile(pd.Series(series), 100)
        self.bandwidth = self.I(lambda: (self.upper - self.lower)/self.upper, name='Bandwidth')
        self.bw_percentile = self.I(calc_percentile, self.bandwidth, name='BW_Percentile')

        # OBV and Swings
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.obv_low = self.I(talib.MIN, self.obv, self.swing_period, name='OBV_Low')
        self.obv_high = self.I(talib.MAX, self.obv, self.swing_period, name='OBV_High')

        # Price Swings
        self.price_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Price_Low')
        self.price_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Price_High')

        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20, name='Vol_MA')
        print("🌙 All indicators aligned with lunar cycles! Ready for trading ✨")

    def next(self):
        if len(self.data.Close) < 100:
            return

        # Moon Dev Debug Prints 🌙
        print(f"🌙 Current BW Percentile: {self.bw_percentile[-1]:.1f}% | Price: {self.data.Close[-1]:.2f} | OBV: {self.obv[-1]:.0f}")

        if not self.position:
            if self.bw_percentile[-1] < self.bandwidth_threshold:
                # Long Entry
                if (self.data.Close[-1] > self.upper[-1] and
                    self._bullish_divergence() and
                    self.data.Volume[-1] > self.vol_ma[-1] * self.volume_multiplier):
                    self._enter_long()

                # Short Entry
                elif (self.data.Close[-1] < self.lower[-1] and
                      self._bearish_divergence() and
                      self.data.Volume[-1] > self.vol_ma[-1] * self.volume_multiplier):
                    self._enter_short()
        else:
            self._check_exits()

    def _bullish_divergence(self):
        price_lows = self.data
```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

# Load and clean data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class SqueezeFlow(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 14
    cmf_period = 20
    bb_period = 20
    bb_dev = 2
    median_period = 100

    def init(self):
        # Bollinger Bands Indicators 🌗
        self.upper_band = self.I(
            lambda close: talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[0],
            self.data.Close
        )
        self.middle_band = self.I(
            lambda close: talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[1],
            self.data.Close
        )
        self.lower_band = self.I(
            lambda close: talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[2],
            self.data.Close
        )

        # ATR for Risk Management 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

        # CMF Calculation Components 🌊
        mfm = ((self.data.Close - self.data.Low) - (self.data.High - self.data.Close)) / (self.data.High - self.data.Low)
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)
        mfv = mfm * self.data.Volume
        self.sum_mfv = self.I(talib.SUM, mfv, self.cmf_period)
        self.sum_vol = self.I(talib.SUM, self.data.Volume, self.cmf_period)

        # Bandwidth Tracking 🌡️
        self.bandwidth_history = []

    def next(self):
        # CMF Calculation 🌊
        current_sum_mfv = self.sum_mfv[-1]
        current_sum_vol = self.sum_vol[-1]
        current_cmf = current_sum_mfv / current_sum_vol if current_sum_vol != 0 else 0
        prev_cmf = (self.sum_mfv[-2]/self.sum_vol[-2]) if len(self.sum_vol) > 1 else 0

        # Bandwidth Calculation 📏
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        middle = self.middle_band[-1]
        bandwidth = (upper - lower) / middle if middle != 0 else 0
        self.bandwidth_history.append(bandwidth)
        if len(self.bandwidth_history) > self.median_period:
            self.bandwidth_history.pop(0)

        # Squeeze Detection 🌀
        squeeze_threshold = 0.5 * np.median(self.bandwidth_history[-self.median_period:]) \
            if len(self.bandwidth_history) >= self.median_period else 1
        in_squeeze = bandwidth < squeeze_threshold

        # Entry Logic 🚀
        if not self.position and in_squeeze and crossover(current_cmf, 0):
            atr = self.atr[-1] if len(self.atr) >= self.atr_period else 0
            if atr > 0:
                size = int(round((self.equity * self.risk_percent) / atr))
                if size > 0:
                    self.buy(size=size, sl=self.data.Close[-1] - atr)
                    print(f"🌙✨ MOONSHOT! Long {size} @ {self.data.Close[-1]:.2f | SL: {self.data.Close[-1] - atr:.2f}")

        # Exit Logic 🛑
        if
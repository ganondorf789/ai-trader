Here's the debugged version of your VolSync Divergence backtest code with Moon Dev themed fixes ✨🌙:

```python
# 🌙 Moon Dev's VolSync Divergence Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# Data Preparation
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolSyncDivergence(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade (fraction of equity)
    atr_period = 14
    bb_period = 20
    swing_period = 5
    
    def init(self):
        # 🌙 Core Indicators
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 20)
        self.kvo = self.I(self._calculate_kvo)
        
        # 🌀 Bollinger Bands Calculation
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(self._calculate_bollinger_bands)
        self.bb_width = self.I(lambda: self.bb_upper - self.bb_lower, name='BB_WIDTH')
        
        # 📏 Volatility Measurements
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # ⚖️ Swing Points Calculation
        self.price_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.price_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.cmf_low = self.I(talib.MIN, self.cmf, self.swing_period)
        self.kvo_low = self.I(talib.MIN, self.kvo, self.swing_period)
        self.cmf_high = self.I(talib.MAX, self.cmf, self.swing_period)
        self.kvo_high = self.I(talib.MAX, self.kvo, self.swing_period)
        
        # 🌪️ Bollinger Squeeze Filter
        self.bb_width_low = self.I(talib.MIN, self.bb_width, 20)
        self.bb_width_high = self.I(talib.MAX, self.bb_width, 20)

    def _calculate_kvo(self):
        kvo = ta.kvo(
            self.data.High, 
            self.data.Low, 
            self.data.Close, 
            self.data.Volume, 
            fast=34, 
            slow=55
        ).iloc[:, 0]
        return kvo

    def _calculate_bollinger_bands(self):
        upper, middle, lower = talib.BBANDS(
            self.data.Close, 
            timeperiod=self.bb_period,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        return upper, middle, lower

    def next(self):
        if len(self.data) < 20 or not self.position.is_empty:
            return

        # 🌗 Bollinger Squeeze Condition
        bb_squeeze = False
        if self.bb_width_high[-1] != self.bb_width_low[-1]:
            squeeze_threshold = self.bb_width_low[-1] + 0.2 * (self.bb_width_high[-1] - self.bb_width_low[-1])
            bb_squeeze = self.bb_width[-1] <= squeeze_threshold

        # 🐂 Bullish Divergence Check
        bull_div = (
            self.price_low[-1] < self.price_low[-2] and
            self.cmf_low[-1] > self.cmf_low[-2] and
            self.kvo_low[-1] > self.kvo_low[-2] and
            self.cmf[-1] > 0 and
            self.data.Close[-1] > self.bb_middle[-1]
        )

        #
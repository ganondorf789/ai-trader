Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLATILITY SQUEEZE TRIGGER STRATEGY 🚀

from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib

# =====================
# 🌌 DATA PREPARATION
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse cosmic dust from column names 🪐
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align celestial coordinates 🌠
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# =====================
# 🌓 STRATEGY CLASS
# =====================
class VolatilitySqueezeTrigger(Strategy):
    # Cosmic configuration 🌌
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    hv_period = 20
    hv_avg_period = 100
    risk_pct = 0.01  # 1% risk per trade
    squeeze_threshold = 0.2  # 20% of average HV

    def init(self):
        # 🌗 Bollinger Bands Formation
        self.mid_band = self.I(talib.SMA, self.data.Close, timeperiod=self.bb_period, name='MID_BAND')
        self.std_dev = self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_period, nbdev=1, name='STD_DEV')
        self.upper_band = self.I(lambda m, s: m + s*self.bb_dev, self.mid_band, self.std_dev, name='UPPER_BAND')
        self.lower_band = self.I(lambda m, s: m - s*self.bb_dev, self.mid_band, self.std_dev, name='LOWER_BAND')
        self.bb_width = self.I(lambda s: s*self.bb_dev*2, self.std_dev, name='BB_WIDTH')

        # 🌑 Volatility Calculations
        roc_ratio = self.I(talib.ROCR, self.data.Close, timeperiod=1, name='ROC_RATIO')
        log_ret = self.I(np.log, roc_ratio, name='LOG_RET')
        self.hv_20 = self.I(talib.STDDEV, log_ret, timeperiod=self.hv_period, name='HV_20')
        self.hv_avg = self.I(talib.SMA, self.hv_20, timeperiod=self.hv_avg_period, name='HV_AVG')

        # 🌠 ATR for Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')

        # 🌕 Trailing Variables
        self.entry_width = None
        self.trail_high = None

    def next(self):
        # 🛸 Skip initial bars without indicator data
        if len(self.data.Close) < max(self.bb_period, self.hv_avg_period) + 10:
            return

        # =====================
        # 🚀 ENTRY LOGIC
        # =====================
        if not self.position:
            # Calculate cosmic squeeze threshold 🌌
            hv_threshold = self.hv_avg[-1] * self.squeeze_threshold
            current_squeeze = self.bb_width[-1] < hv_threshold
            
            # Check planetary alignment conditions 🌑🌕🌑
            prev_in_band = (self.lower_band[-2] < self.data.Close[-2] < self.upper_band[-2])
            breakout_trigger = (self.data.Close[-1] > self.upper_band[-1])
            
            if current_squeeze and prev_in_band and breakout_trigger:
                # 🌙 Calculate Moon Position Size
                atr_val = self.atr[-1]
                stop_loss = self.data.Close[-1] - 2*atr_val
                risk_per_share = self.data.Close
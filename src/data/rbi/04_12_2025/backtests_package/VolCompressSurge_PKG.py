I'll fix the code by removing all backtesting.lib imports and replacing any crossover/crossunder functions with manual implementations. Here's the corrected version:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLCOMPRESSSURGE STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from scipy.stats import percentileofscore

# =====================
# DATA PREPARATION 🌐
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping and resampling to 2H timeframe 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime').resample('2H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# =====================
# STRATEGY CLASS 🚀🌙
# =====================
class VolCompressSurge(Strategy):
    bb_period = 20
    bb_std = 2
    kc_period = 20
    kc_atr_mult = 1.5
    lookback_period = 100
    risk_pct = 0.01
    max_risk_pct = 0.05

    def init(self):
        # =====================
        # INDICATORS CALCULATION 🌗
        # =====================
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands 🌗
        self.bb_upper = self.I(talib.BBANDS, close, 
                              timeperiod=self.bb_period, 
                              nbdevup=self.bb_std, 
                              nbdevdn=self.bb_std, 
                              matype=0, name='BB_UPPER')[0]
        self.bb_lower = self.I(talib.BBANDS, close, 
                              timeperiod=self.bb_period, 
                              nbdevup=self.bb_std, 
                              nbdevdn=self.bb_std, 
                              matype=0, name='BB_LOWER')[2]
        
        # BB Width Percentile 🌓
        bb_width = (self.bb_upper - self.bb_lower)/self.I(talib.SMA, close, 20)
        self.bb_width_pct = self.I(lambda x: x.rolling(self.lookback_period).apply(
            lambda s: percentileofscore(s, s[-1]) - 0.0001,  # Fix pandas 1.3+ issue
            bb_width, name='BB_WIDTH_PCT')
        
        # Volume Percentile 🌊
        volume = self.data.Volume
        self.vol_pct = self.I(lambda x: x.rolling(self.lookback_period).apply(
            lambda s: percentileofscore(s, s[-1]) - 0.0001,
            volume, name='VOL_PCT')
        
        # Keltner Channels 🌗
        self.atr = self.I(talib.ATR, high, low, close, self.kc_period, name='ATR')
        self.ema = self.I(talib.EMA, close, self.kc_period, name='EMA')
        self.kc_upper = self.I(lambda: self.ema + self.atr * self.kc_atr_mult, name='KC_UP')
        self.kc_lower = self.I(lambda: self.ema - self.atr * self.kc_atr_mult, name='KC_LO')

    def next(self):
        # =====================
        # RISK MANAGEMENT 🛡️
        # =====================
        equity = self.equity
        price = self.data.Close[-1]
        
        if self.position:
            # =====================
            # EXIT LOGIC 🚪
            # =====================
            if self.position.is_long:
                if price >= self.kc_upper[-1]:
                    self.position.close()
                    print(f"🌕 MOON EXIT LONG | Price {
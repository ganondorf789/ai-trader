Here's the complete fixed backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 Moon Dev's VortexDivergence Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import numpy as np

# ---------------------------
# DATA PREPARATION
# ---------------------------
print("🌕 Initializing Moon Dev's Vortex Divergence Strategy...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Map to backtesting.py format 🗺️
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert index to datetime ⏳
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime').sort_index()

# ---------------------------
# STRATEGY CLASS
# ---------------------------
class VortexDivergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    vortex_period = 14
    cmo_period = 14
    volume_ma_period = 20
    swing_period = 20
    keltner_ema = 20
    keltner_atr = 20
    atr_multiplier = 2

    def init(self):
        print("🌖 Initializing indicators...")
        # 🌪️ Vortex Indicator
        self.vi_plus = self.I(ta.vortex, 
                            self.data.High, 
                            self.data.Low, 
                            self.data.Close, 
                            self.vortex_period,
                            name='VI_PLUS',
                            col_names=['VIp_14'])
        
        self.vi_minus = self.I(ta.vortex, 
                             self.data.High, 
                             self.data.Low, 
                             self.data.Close, 
                             self.vortex_period,
                             name='VI_MINUS',
                             col_names=['VIm_14'])

        # 📈 Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period)

        # 📊 Volume Confirmation
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)

        # 🛑 Swing High/Low for Stops
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)

        # 📉 Keltner Channels
        self.ema = self.I(talib.EMA, self.data.Close, self.keltner_ema)
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         self.keltner_atr)

    def next(self):
        # 🌙 Moon Dev Debug Checks
        if len(self.data) < max(self.vortex_period, 
                              self.cmo_period, 
                              self.volume_ma_period, 
                              self.swing_period, 
                              self.keltner_ema) + 1:
            print("🌑 Not enough data yet - waiting for full indicator periods")
            return

        # 🔄 Vortex Crossover Logic
        vi_cross_up = (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1])
        vi_cross_down = (self.vi_minus[-2] < self.vi_plus[-2]) and (self.vi_minus[-1] > self.vi_plus[-1])

        # 📉 Momentum Divergence Check
        current_high, prev_high = self.data.High[-1], self.data.High[-2]
        current_low, prev_low = self.data.Low[-1], self.data.Low[-2]
        current_cmo, prev_cmo = self.cmo[-1], self.cmo[-2]
        
        bearish_divergence = (current_high > prev_high) and (current_cmo < prev_cmo)
        bullish_divergence = (current_low < prev_low) and
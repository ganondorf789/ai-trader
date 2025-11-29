I'll fix the code by removing all `backtesting.lib` imports and replacing any related functions. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's SqueezeDivergence Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pytz
from datetime import time

class SqueezeDivergence(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    bb_period = 20
    kc_period = 20
    kc_mult = 1.5
    swing_period = 20
    
    def init(self):
        # 🌙 Clean and prepare data
        self.data.df.index = pd.to_datetime(self.data.df.index).tz_convert('US/Eastern')
        
        # 🌟 Core Indicators
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, self.bb_period, 2, 2, name='BB_UPPER')[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, self.bb_period, 2, 2, name='BB_MIDDLE')[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, self.bb_period, 2, 2, name='BB_LOWER')[2]
        
        # Keltner Channels
        self.ema = self.I(talib.EMA, self.data.Close, self.kc_period, name='EMA')
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.kc_period, name='ATR_KC')
        self.kc_upper = self.I(lambda: self.ema + self.kc_mult * self.atr_kc, name='KC_UPPER')
        self.kc_lower = self.I(lambda: self.ema - self.kc_mult * self.atr_kc, name='KC_LOWER')
        
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        # Liquidation Zones
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='SWING_LOW')
        
        # Volume indicator
        self.volume_ema = self.I(talib.EMA, self.data.Volume, 20, name='VOL_EMA')
        
        # 🌙 Moon Dev Debug: Indicators initialized successfully!
        print("🌙✨ Moon Dev Indicators Activated! All systems go! ✨🌙")
        
    def next(self):
        # 🌙 Skip if not enough data
        if len(self.data) < 50:
            return
        
        # 🌟 Current Values
        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        time = self.data.index[-1].time()
        
        # 🚀 Entry Conditions
        # 1. Bollinger Squeeze
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_middle[-1]
        bb_widths = [self.bb_upper[i]-self.bb_lower[i]/self.bb_middle[i] for i in range(-20,0)]
        squeeze = bb_width < np.percentile(bb_widths, 20)
        
        # 2. Volume Divergence
        vol_div = (high > self.data.High[-2]) and (self.volume_ema[-1] < self.volume_ema[-2])
        
        # 3. Liquidity Cluster
        liq_cluster = any(abs(close - lvl)/close < 0.005 
                         for lvl in [self.swing_high[-1], self.swing_low[-1]])
        
        # 4. Keltner Break
        trend_up = close > self.ema[-1]
        kc_break_short = close < self.kc_lower[-1] and trend_up
        kc_break_long = close > self.kc_upper[-1] and not trend_up
        
        # 5. Time Filter
        session_time = time >= time(9
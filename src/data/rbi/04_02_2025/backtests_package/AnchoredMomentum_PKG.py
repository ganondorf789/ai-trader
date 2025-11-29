Here's the fixed code with all backtesting.lib references removed and proper indicator implementations:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class AnchoredMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    swing_period = 20
    atr_period = 14
    volume_ma_period = 5
    
    def init(self):
        # Moon Dev debug print 🌙
        print("✨ Initializing Anchored Momentum Strategy with Moon Dev's Package AI ✨")
        
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)
        
        # Heikin-Ashi Calculation
        ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        ha_open = pd.Series(index=self.data.df.index, dtype=float)
        ha_open.iloc[0] = self.data.Open.iloc[0]
        for i in range(1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        ha_high = self.data.High.combine(ha_open.combine(ha_close, max), max)
        ha_low = self.data.Low.combine(ha_open.combine(ha_close, min), min)
        
        self.ha_open = self.I(lambda: ha_open.values, name='HA_Open')
        self.ha_high = self.I(lambda: ha_high.values, name='HA_High')
        self.ha_low = self.I(lambda: ha_low.values, name='HA_Low')
        self.ha_close = self.I(lambda: ha_close.values, name='HA_Close')
        
        # Swing Points
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='Swing_Low')
        
        # Anchored VWAP Calculation
        hlc3 = (self.data.High + self.data.Low + self.data.Close) / 3
        tpv = hlc3 * self.data.Volume
        cumsum_tpv = np.cumsum(tpv)
        cumsum_vol = np.cumsum(self.data.Volume)
        
        swing_highs = self.data.High == self.swing_high
        swing_lows = self.data.Low == self.swing_low
        
        last_swing_high_idx = np.zeros(len(self.data), dtype=int)
        current_anchor = 0
        for i in range(len(self.data)):
            if swing_highs[i]:
                current_anchor = i
            last_swing_high_idx[i] = current_anchor
            
        last_swing_low_idx = np.zeros(len(self.data), dtype=int)
        current_anchor = 0
        for i in range(len(self.data)):
            if swing_lows[i]:
                current_anchor = i
            last_swing_low_idx[i] = current_anchor
        
        anchored_vwap_long = np.zeros(len(self.data))
        anchored_vwap_short = np.zeros(len(self.data))
        for i in range(len(self.data)):
            start_long = last_swing_low_idx[i]
            start_short = last_swing_high_idx[i]
            
            vwap_long = (cumsum_tpv[i] - (cumsum_tpv[start_long-1] if start_long >0 else 0)) / \
                       (cumsum_vol[i] - (cumsum_vol[start_long-1] if start_long >0 else 0))
            vwap_short = (cumsum_tpv[i] - (cumsum_tpv[start_short-1] if start_short >0 else 0)) / \
                        (cumsum_vol[i] - (cumsum_vol[start_short-1] if start_short >0 else 0))
            
            anchored_vwap_long[i] = vwap_long
            anchored_vwap_short[i] = vwap_short
        
        self.anchored_vwap_long = self.I(lambda: anchored_vwap_long, name='Anchored_VWAP_Long')
        self.anchored_vwap_short = self.I(lambda: anchored_vwap_short, name='Anchored_VWAP_Short')
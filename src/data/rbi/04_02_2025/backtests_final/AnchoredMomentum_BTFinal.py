Here's the fixed backtest code with all technical issues resolved while preserving the original strategy logic:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class AnchoredMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (percentage of equity)
    swing_period = 20
    atr_period = 14
    volume_ma_period = 5
    
    def init(self):
        # Moon Dev debug print 🌙
        print("✨ Initializing Anchored Momentum Strategy with Moon Dev's Package AI ✨")
        
        # Clean and prepare data
        self.data.df.columns = [col.strip().lower() for col in self.data.df.columns]
        self.data.df = self.data.df.loc[:, ~self.data.df.columns.str.contains('unnamed')]
        
        # Heikin-Ashi Calculation
        ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        ha_open = np.zeros(len(self.data))
        ha_open[0] = self.data.Open[0]
        for i in range(1, len(ha_open)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high = np.maximum(np.maximum(self.data.High, ha_open), ha_close)
        ha_low = np.minimum(np.minimum(self.data.Low, ha_open), ha_close)
        
        self.ha_open = self.I(ha_open, name='HA_Open')
        self.ha_high = self.I(ha_high, name='HA_High')
        self.ha_low = self.I(ha_low, name='HA_Low')
        self.ha_close = self.I(ha_close, name='HA_Close')
        
        # Swing Points
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='Swing_Low')
        
        # ATR for position sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
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
            
            vwap_long = (cumsum_tpv[i] - (cumsum_tpv[start_long-1] if start_long > 0 else 0)) / \
                       (cumsum_vol[i] - (cumsum_vol[start_long-1] if start_long > 0 else 0))
            vwap_short = (cumsum_tpv[i] - (cumsum_tpv[start_short-1] if start_short > 0 else 0)) / \
                        (cumsum_vol[i] - (cumsum_vol[start_short-1] if start_short > 0 else 0))
            
            anchored_vwap_long[i] = vwap_long
            anchored_vwap_short[i] = vwap_short
        
        self.anchored_vwap_long = self.I(anchored_vwap_long, name='Anchored_VWAP_Long')
        self.anchored_v